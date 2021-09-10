import matplotlib
matplotlib.use('Agg')

import argparse
import json
import logging
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dateutil.relativedelta import relativedelta
from shutil import copyfile

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from daft_exprt.data_loader import prepare_data_loaders
from daft_exprt.extract_features import FEATURES_HPARAMS, check_features_config_used
from daft_exprt.generate import extract_reference_parameters, prepare_sentences_for_inference, generate_mel_specs
from daft_exprt.hparams import HyperParams
from daft_exprt.logger import DaftExprtLogger
from daft_exprt.loss import DaftExprtLoss
from daft_exprt.model import DaftExprt
from daft_exprt.utils import get_nb_jobs


_logger = logging.getLogger(__name__)


def check_train_config(hparams):
    ''' Check hyper-parameters used for training are the same than the one used to extract features

    :param hparams:         hyper-parameters currently used for training
    '''
    # extract features dirs used for training
    with open(hparams.training_files, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    features_dirs = [line.strip().split(sep='|')[0] for line in lines]
    features_dirs = list(set(features_dirs))
    
    # compare hyper-params
    _logger.info('--' * 30)
    _logger.info(f'Comparing training config with the one used to extract features'.upper())
    for features_dir in features_dirs:
        same_config = check_features_config_used(features_dir, hparams)
        assert(same_config), _logger.error(f'Parameters used for feature extraction in "{features_dir}" '
                                           f'mismatch with current training parameters.')
    _logger.info('--' * 30 + '\n')


def save_checkpoint(model, optimizer, hparams, learning_rate,
                    iteration, best_val_loss=None, filepath=None):
    ''' Save a model/optimizer state and store additional training info

    :param model:               current model state
    :param optimizer:           current optimizer state
    :param hparams:             hyper-parameters used for training
    :param learning_rate:       current learning rate value
    :param iteration:           current training iteration
    :param best_val_loss:       current best validation loss
    :param filepath:            path to save the checkpoint
    '''
    # get output directory where checkpoint is saved and make directory if it doesn't exists
    output_directory = os.path.dirname(filepath)
    os.makedirs(output_directory, exist_ok=True)
    # save checkpoint
    _logger.info(f'Saving model and optimizer state at iteration "{iteration}" to "{filepath}"')
    torch.save({'iteration': iteration,
                'learning_rate': learning_rate,
                'best_val_loss': best_val_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config_params': hparams.__dict__.copy()}, filepath)


def load_checkpoint(checkpoint_path, gpu, model, optimizer, hparams):
    ''' Load a model/optimizer state and additional training info

    :param checkpoint_path:     path of the checkpoint to load
    :param gpu:                 GPU ID that hosts the model
    :param model:               current model state we want to update with checkpoint
    :param optimizer:           current optimizer state we want to update with checkpoint
    :param hparams:             hyper-parameters used for training

    :return: model/optimizer and additional training info
    '''
    # load checkpoint dict
    # map model to be loaded to specified single gpu
    assert os.path.isfile(checkpoint_path), \
        _logger.error(f'Checkpoint "{checkpoint_path}" does not exist')
    _logger.info(f'Loading checkpoint "{checkpoint_path}"')
    checkpoint_dict = torch.load(checkpoint_path, map_location=f'cuda:{gpu}')
    # compare current hparams with the ones used in checkpoint
    hparams_checkpoint = HyperParams(verbose=False, **checkpoint_dict['config_params'])
    params_to_compare = hparams.__dict__.copy()
    for param in params_to_compare:
        if param in FEATURES_HPARAMS:
            assert(getattr(hparams, param) == getattr(hparams_checkpoint, param)), \
                _logger.error(f'Parameter "{param}" is different between current config and the one used in checkpoint -- '
                              f'Was {getattr(hparams_checkpoint, param)} in checkpoint and now is {getattr(hparams, param)}')
        else:
            if not hasattr(hparams, param):
                _logger.warning(f'Parameter "{param}" does not exist in the current training config but existed in checkpoint config')
            elif not hasattr(hparams_checkpoint, param):
                _logger.warning(f'Parameter "{param}" exists in the current training confid but did not exist in checkpoint config')
            elif getattr(hparams, param) != getattr(hparams_checkpoint, param):
                _logger.warning(f'Parameter "{param}" has changed -- Was {getattr(hparams_checkpoint, param)} '
                                f'in checkpoint and now is {getattr(hparams, param)}')
        
    # assign checkpoint weights to the model
    try:
        model.load_state_dict(checkpoint_dict['state_dict'])
    except RuntimeError as e:
        _logger.error(f'Error when trying to load the checkpoint -- "{e}"\n')
    
    # check if the optimizers are compatible
    k_new = optimizer.param_groups
    k_loaded = checkpoint_dict['optimizer']['param_groups']
    if len(k_loaded) != len(k_new):
        _logger.warning(f'The optimizer in the loaded checkpoint does not have the same number of parameters '
                        f'as the blank optimizer -- Creating a new optimizer.')
    else:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    
    # load additional values
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    best_val_loss = checkpoint_dict['best_val_loss']
    _logger.info(f'Loaded checkpoint "{checkpoint_path}" from iteration "{iteration}"\n')

    return model, optimizer, iteration, learning_rate, best_val_loss


def update_learning_rate(hparams, iteration):
    ''' Increase the learning rate linearly for the first warmup_steps training steps,
        and decrease it thereafter proportionally to the inverse square root of the step number
    '''
    initial_learning_rate = hparams.initial_learning_rate
    max_learning_rate = hparams.max_learning_rate
    warmup_steps = hparams.warmup_steps
    if iteration < warmup_steps:
        learning_rate = (max_learning_rate - initial_learning_rate) / warmup_steps * iteration + initial_learning_rate
    else:
        learning_rate = iteration ** -0.5 * max_learning_rate / warmup_steps ** -0.5
    
    return learning_rate


def generate_benchmark_sentences(model, hparams, output_dir):
    ''' Generate benchmark sentences using Daft-Exprt model
    
    :param model:           model to use for synthesis
    :param hparams:         hyper-params used for training/synthesis
    :param output_dir:      directory to store synthesized files
    '''
    # set random speaker id
    speaker_id = random.choice(hparams.speakers_id)
    # choose reference for style transfer
    with open(hparams.validation_files, 'r', encoding='utf-8') as f:
        references = [line.strip().split('|') for line in f]
    reference = random.choice(references)
    reference_path, file_name = reference[0], reference[1]
    speaker_name = [speaker for speaker in hparams.speakers if reference_path.endswith(speaker)][0]
    audio_ref = f'{os.path.join(hparams.data_set_dir, speaker_name, "wavs", file_name)}.wav'
    # display info
    _logger.info('\nGenerating benchmark sentences with the following parameters:')
    _logger.info(f'speaker_id = {speaker_id}')
    _logger.info(f'audio_ref = {audio_ref}\n')
    
    # prepare benchmark sentences
    n_jobs = get_nb_jobs('max')
    text_file = os.path.join(hparams.benchmark_dir, hparams.language, 'sentences.txt')
    sentences, file_names = \
        prepare_sentences_for_inference(text_file, output_dir, hparams, n_jobs)
    # extract reference prosody parameters
    extract_reference_parameters(audio_ref, output_dir, hparams)
    # duplicate reference parameters
    file_name = os.path.basename(audio_ref).replace('.wav', '')
    refs = [os.path.join(output_dir, f'{file_name}.npz') for _ in range(len(sentences))]
    # generate mel_specs and audios with Griffin-Lim
    speaker_ids = [speaker_id for _ in range(len(sentences))]
    generate_mel_specs(model, sentences, file_names, speaker_ids, refs,
                       output_dir, hparams, use_griffin_lim=True)
    # copy audio ref
    copyfile(audio_ref, os.path.join(output_dir, f'{file_name}.wav'))


def validate(gpu, model, criterion, val_loader, hparams):
    ''' Handles all the validation scoring and printing

    :param gpu:             GPU ID that hosts the model
    :param model:           model to evaluate
    :param criterion:       criterion used for training
    :param val_loader:      validation Data Loader
    :param hparams:         hyper-params used for training

    :return: validation loss score
    '''
    # initialize variables
    val_loss = 0.
    val_indiv_loss = {
        'duration_loss': 0., 'energy_loss':0., 'pitch_loss': 0.,
        'mel_spec_l1_loss': 0., 'mel_spec_l2_loss': 0.
    }
    val_targets, val_outputs = [], []
    
    # set eval mode
    model.eval()
    with torch.no_grad():
        # iterate over validation set
        for i, batch in enumerate(val_loader):
            if hparams.multiprocessing_distributed:
                inputs, targets, _ = model.module.parse_batch(gpu, batch)
            else:
                inputs, targets, _ = model.parse_batch(gpu, batch)
            outputs = model(inputs)
            loss, individual_loss = criterion(outputs, targets, iteration=0)
            val_targets.append(targets)
            val_outputs.append(outputs)
            val_loss += loss.item()
            for key in val_indiv_loss:
                val_indiv_loss[key] += individual_loss[key]
        # normalize losses
        val_loss = val_loss / (i + 1)
        for key in val_indiv_loss:
            val_indiv_loss[key] = val_indiv_loss[key] / (i + 1)

    return val_loss, val_indiv_loss, val_targets, val_outputs


def train(gpu, hparams, log_file):
    ''' Train Daft-Exprt model
    
    :param gpu:         GPU ID to host the model
    :param hparams:     hyper-params used for training
    :param log_file:    file path for logging
    '''
    # ---------------------------------------------------------
    # initialize distributed group
    # ---------------------------------------------------------
    if hparams.multiprocessing_distributed:
        # for multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        hparams.rank = hparams.rank * hparams.ngpus_per_node + gpu
        dist.init_process_group(backend=hparams.dist_backend, init_method=hparams.dist_url,
                                world_size=hparams.world_size, rank=hparams.rank)
    
    # ---------------------------------------------------------
    # create loggers
    # ---------------------------------------------------------
    # set logger config
    # we log INFO to file only from rank0, node0 to avoid unnecessary log duplication
    if hparams.rank == 0:
        logging.basicConfig(
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ],
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO
        )
        # create tensorboard logger
        log_dir = os.path.dirname(log_file)
        tensorboard_logger = DaftExprtLogger(log_dir)
    else:
        logging.basicConfig(
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ],
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.ERROR
        )
    
    # ---------------------------------------------------------
    # create model
    # ---------------------------------------------------------
    # load model on GPU
    torch.cuda.set_device(gpu)
    model = DaftExprt(hparams).cuda(gpu)
    
    # for multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices
    if hparams.multiprocessing_distributed:
        model = DDP(model, device_ids=[gpu])
    
    # ---------------------------------------------------------
    # define training loss and optimizer
    # ---------------------------------------------------------
    criterion = DaftExprtLoss(gpu, hparams)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     betas=hparams.betas, eps=hparams.epsilon,
                     weight_decay=hparams.weight_decay, amsgrad=False)
    
    # ---------------------------------------------------------
    # optionally resume from a checkpoint
    # ---------------------------------------------------------
    iteration, best_val_loss = 1, float('inf')
    if hparams.checkpoint != "":
        model, optimizer, iteration, learning_rate, best_val_loss = \
            load_checkpoint(hparams.checkpoint, gpu, model, optimizer, hparams)
        iteration += 1  # next iteration is iteration + 1
    
    # ---------------------------------------------------------
    # set learning rate
    # ---------------------------------------------------------
    learning_rate = update_learning_rate(hparams, iteration)
    for param_group in optimizer.param_groups:
        if param_group['lr'] is not None:
            param_group['lr'] = learning_rate
    
    # ---------------------------------------------------------
    # prepare Data Loaders
    # ---------------------------------------------------------
    train_loader, train_sampler, val_loader, nb_training_examples = \
        prepare_data_loaders(hparams, num_workers=8)
    
    # ---------------------------------------------------------
    # display training info
    # ---------------------------------------------------------
    # compute the number of epochs
    nb_iterations_per_epoch = int(len(train_loader) / hparams.accumulation_steps)
    epoch_offset = max(0, int(iteration / nb_iterations_per_epoch))
    epochs = int(hparams.nb_iterations / nb_iterations_per_epoch) + 1

    _logger.info('**' * 40)
    _logger.info(f"Batch size: {hparams.batch_size * hparams.accumulation_steps * hparams.world_size:_}")
    _logger.info(f"Nb examples: {nb_training_examples:_}")
    _logger.info(f"Nb iterations per epoch: {nb_iterations_per_epoch:_}")
    _logger.info(f"Nb total of epochs: {epochs:_}")
    _logger.info(f"Started at epoch: {epoch_offset:_}")
    _logger.info('**' * 40 + '\n')

    # =========================================================
    #                   MAIN TRAINNIG LOOP
    # =========================================================
    # set variables
    tot_loss = 0.
    indiv_loss = {
        'speaker_loss': 0., 'post_mult_loss': 0.,
        'duration_loss': 0., 'energy_loss':0., 'pitch_loss': 0.,
        'mel_spec_l1_loss': 0., 'mel_spec_l2_loss': 0.
    }
    total_time = 0.
    start = time.time()
    accumulation_step = 0
    
    model.train()  # set training mode
    model.zero_grad()  # set gradients to 0
    for epoch in range(epoch_offset, epochs):
        _logger.info(30 * '=')
        _logger.info(f"| Epoch: {epoch}")
        _logger.info(30 * '=' + '\n')
        
        # shuffle dataset
        if hparams.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)
        
        # iterate over examples
        for batch in train_loader:
            # ---------------------------------------------------------
            # forward pass
            # ---------------------------------------------------------
            if hparams.multiprocessing_distributed:
                inputs, targets, _ = model.module.parse_batch(gpu, batch)
            else:
                inputs, targets, _ = model.parse_batch(gpu, batch)
            
            outputs = model(inputs)
            loss, individual_loss = criterion(outputs, targets, iteration)  # loss / batch_size
            loss = loss / hparams.accumulation_steps  # loss / (batch_size * accumulation_steps)
            
            # track losses
            tot_loss += loss.item()
            for key in individual_loss:
                # individual losses are already detached from the graph
                # individual_loss / (batch_size * accumulation_steps)
                indiv_loss[key] += individual_loss[key] / hparams.accumulation_steps

            # ---------------------------------------------------------
            # backward pass
            # ---------------------------------------------------------
            loss.backward()
            accumulation_step += 1

            # ---------------------------------------------------------
            # accumulate gradient
            # ---------------------------------------------------------
            if accumulation_step == hparams.accumulation_steps:
                # clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
                # update weights
                optimizer.step()

                # ---------------------------------------------------------
                # reporting
                # ---------------------------------------------------------
                if not math.isnan(tot_loss):
                    if hparams.rank == 0:
                        # get current learning rate
                        for param_group in optimizer.param_groups:
                            if param_group['lr'] is not None:
                                learning_rate = param_group['lr']
                                break
                        # display iteration stats
                        duration = time.time() - start
                        total_time += duration
                        _logger.info(f'Train loss [{iteration}]: {tot_loss:.6f} Grad Norm {grad_norm:.6f} '
                                    f'{duration:.2f}s/it (LR {learning_rate:.6f})')
                        # update tensorboard logging
                        tensorboard_logger.log_training(tot_loss, indiv_loss, grad_norm,
                                                        learning_rate, duration, iteration)
                    # barrier for distributed processes
                    if hparams.multiprocessing_distributed:
                        dist.barrier()

                # ---------------------------------------------------------
                # model evaluation
                # ---------------------------------------------------------
                if iteration % hparams.iters_check_for_model_improvement == 0:
                    # validate model
                    _logger.info('Validating....')
                    val_loss, val_indiv_loss, val_targets, val_outputs = validate(gpu, model, criterion, val_loader, hparams)
                    if hparams.rank == 0:
                        # display remaining time
                        _logger.info(f"Validation loss {iteration}: {val_loss:.6f} ")
                        _logger.info("estimated required time = {0.days:02}:{0.hours:02}:{0.minutes:02}:{0.seconds:02}".
                                     format(relativedelta(seconds=int((hparams.nb_iterations - iteration) *
                                            (total_time / hparams.iters_check_for_model_improvement)))))
                        total_time = 0
                        # log validation loss
                        tensorboard_logger.log_validation(val_loss, val_indiv_loss, val_targets,
                                                          val_outputs, model, hparams, iteration)

                        # save as the best model
                        if val_loss < best_val_loss:
                            # update validation loss
                            _logger.info('Congrats!!! A new best model. You are the best!')
                            best_val_loss = val_loss
                            # save checkpoint and generate benchmark sentences
                            checkpoint_path = os.path.join(hparams.output_directory, 'checkpoints', 'DaftExprt_best')
                            save_checkpoint(model, optimizer, hparams, learning_rate,
                                            iteration, best_val_loss, checkpoint_path)
                            output_dir = os.path.join(hparams.output_directory, 'checkpoints', 'best_checkpoint')
                            generate_benchmark_sentences(model, hparams, output_dir)
                    # barrier for distributed processes
                    if hparams.multiprocessing_distributed:
                        dist.barrier()

                # ---------------------------------------------------------
                # save the model
                # ---------------------------------------------------------
                if iteration % hparams.iters_per_checkpoint == 0:
                    if hparams.rank == 0:
                        checkpoint_path = os.path.join(hparams.output_directory, 'checkpoints', f'DaftExprt_{iteration}')
                        save_checkpoint(model, optimizer, hparams, learning_rate,
                                        iteration, best_val_loss, checkpoint_path)
                        output_dir = os.path.join(hparams.output_directory, 'checkpoints', f'chk_{iteration}')
                        generate_benchmark_sentences(model, hparams, output_dir)
                    # barrier for distributed processes
                    if hparams.multiprocessing_distributed:
                        dist.barrier()
                
                # ---------------------------------------------------------
                # reset variables
                # ---------------------------------------------------------
                iteration += 1
                tot_loss = 0.
                indiv_loss = {
                    'speaker_loss': 0., 'post_mult_loss': 0.,
                    'duration_loss': 0., 'energy_loss':0., 'pitch_loss': 0.,
                    'mel_spec_l1_loss': 0., 'mel_spec_l2_loss': 0.
                }
                start = time.time()
                accumulation_step = 0
                
                model.train()  # set training mode
                model.zero_grad()  # set gradients to 0
                
                # ---------------------------------------------------------
                # adjust learning rate
                # ---------------------------------------------------------
                learning_rate = update_learning_rate(hparams, iteration)
                for param_group in optimizer.param_groups:
                    if param_group['lr'] is not None:
                        param_group['lr'] = learning_rate


def launch_training(data_set_dir, config_file, benchmark_dir, log_file, world_size=1, rank=0,
                    multiprocessing_distributed=True, master='tcp://localhost:54321'):
    ''' Launch training in distributed mode or on a single GPU
        PyTorch distributed training is performed using DistributedDataParrallel API
        Inspired from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    
        - multiprocessing_distributed=False:
            Training is performed using only GPU 0 on the machine
        
        - multiprocessing_distributed=True:
            Multi-processing distributed training is performed with DistributedDataParrallel API.
            X distributed processes are launched on the machine, with X the total number of GPUs
            on the machine. Each process replicates the same model to a unique GPU, and each GPU
            consumes a different partition of the input data. DistributedDataParrallel takes care
            of gradient averaging and model parameter update on all GPUs. This is the go-to method
            when model can fit on one GPU card.
            - world_size=1:
                One machine is used for distributed training. The machine launches X distributed processes.
            - world_size=N:
                N machines are used for distributed training. Each machine launches X distributed processes.
    '''
    # set logger config
    if rank == 0:
        logging.basicConfig(
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ],
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO
        )
    else:
        logging.basicConfig(
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ],
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.ERROR
        )
    
    # get hyper-parameters
    with open(config_file) as f:
        data = f.read()
    config = json.loads(data)
    hparams = HyperParams(verbose=False, **config)
    
    # count number of GPUs on the machine
    ngpus_per_node = torch.cuda.device_count()
    
    # set default values
    if multiprocessing_distributed:
        hparams.dist_url = f'{master}'
        # since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        # here we assume that each node has the same number of GPUs
        world_size = ngpus_per_node * world_size
    else:
        rank, gpu = 0, 0
    
    # update hparams
    hparams.data_set_dir = data_set_dir
    hparams.config_file = config_file
    hparams.benchmark_dir = benchmark_dir
    
    hparams.rank = rank
    hparams.world_size = world_size
    hparams.ngpus_per_node = ngpus_per_node
    hparams.multiprocessing_distributed = multiprocessing_distributed
    
    # check that config used for training is the same than the one used for features extraction
    check_train_config(hparams)
    # save hyper-params to config.json
    if rank == 0:
        hparams.save_hyper_params(hparams.config_file)
    
    # check if multiprocessing distributed is deactivated but feasible
    if not multiprocessing_distributed and ngpus_per_node > 1:
        _logger.warning(f'{ngpus_per_node} GPUs detected but distributed training is not set. '
                        f'Training on only 1 GPU.\n')
    
    # define cudnn variables
    torch.manual_seed(0)
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    torch.backends.cudnn.deterministic = hparams.cudnn_deterministic
    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        torch.backends.cudnn.deterministic = True
        _logger.warning('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! You may see unexpected behavior when '
                        'restarting from checkpoints.\n')
    
    # display training setup info
    _logger.info(f'PyTorch version -- {torch.__version__}')
    _logger.info(f'CUDA version -- {torch.version.cuda}')
    _logger.info(f'CUDNN version -- {torch.backends.cudnn.version()}')
    _logger.info(f'CUDNN enabled = {torch.backends.cudnn.enabled}')
    _logger.info(f'CUDNN deterministic = {torch.backends.cudnn.deterministic}')
    _logger.info(f'CUDNN benchmark = {torch.backends.cudnn.benchmark}\n')
    
    # clear handlers
    _logger.handlers.clear()
    
    # launch multi-processing distributed training
    if multiprocessing_distributed:
        # use torch.multiprocessing.spawn to launch distributed processes
        mp.spawn(train, nprocs=ngpus_per_node, args=(hparams, log_file))
    # simply call train function
    else:
        train(gpu, hparams, log_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch N processes per '
                             'node, which has N GPUs. This is the fastest way to use PyTorch for '
                             'either single node or multi node data parallel training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int, default=0,
                        help='node rank for distributed training')
    parser.add_argument('--master', type=str, default='tcp://localhost:54321',
                        help='url used to set up distributed training')
    parser.add_argument('--data_set_dir', type=str, required=True,
                        help='Data set containing .wav files')
    parser.add_argument('--config_file', type=str, required=True,
                        help='JSON configuration file to initialize hyper-parameters for training')
    parser.add_argument('--benchmark_dir', type=str, required=True,
                        help='directory to load benchmark sentences')
    parser.add_argument('--log_file', type=str, required=True,
                        help='path to save logger outputs')

    args = parser.parse_args()
    
    # launch training
    launch_training(args.data_set_dir, args.config_file, args.benchmark_dir, args.log_file,
                    args.world_size, args.rank, args.multiprocessing_distributed, args.master)
