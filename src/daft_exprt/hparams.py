import json
import logging
import os
import sys

from pathlib import Path

from daft_exprt.symbols import pad, symbols_english


_logger = logging.getLogger(__name__)


'''
    Hyper-parameters used for pre-processing and training
'''


class HyperParams(object):
    def __init__(self, verbose=True, **kwargs):
        ''' Initialize hyper-parameter values for data pre-processing and training

        :param verbose:         whether to display logger info/warnings or not
        :param kwargs:          keyword arguments to modify hyper-params values
        '''
        # display some logger info
        if verbose:
            _logger.info('--' * 30)
            _logger.info('Setting Hyper-Parameters'.upper())
            _logger.info('--' * 30)

        ###########################################
        #### hard-coded hyper-parameter values ####
        ###########################################
        # misc hyper-parameters
        self.minimum_wav_duration = 1000  # minimum duration (ms) of the audio files used for training
        
        # mel-spec extraction hyper-parameters
        self.centered = True  # extraction window is centered on the time step when doing FFT
        self.min_clipping = 1e-5  # min clipping value when creating mel-specs
        self.sampling_rate = 22050  # sampling rate of the audios in the data set
        self.mel_fmin = 0  # lowest frequency (in Hz) of the mel-spectrogram
        self.mel_fmax = 8000  # highest frequency (in Hz) of the mel-spectrogram
        self.n_mel_channels = 80  # number of mel bands to generate
        self.filter_length = 1024  # FFT window length (in samples)
        self.hop_length = 256  # length (in samples) between successive analysis windows for FFT
        
        # REAPER pitch extraction hyper-parameters
        self.f0_interval = 0.005
        self.min_f0 = 40
        self.max_f0 = 500
        self.uv_interval = 0.01
        self.uv_cost = 0.9
        self.order = 1
        self.cutoff = 25
        
        # training hyper-parameters
        self.seed = 1234  # seed used to initialize weights
        self.cudnn_enabled = True  # parameter used when initializing training
        self.cudnn_benchmark = False  # parameter used when initializing training
        self.cudnn_deterministic = True  # parameter used when initializing training
        self.dist_backend = 'nccl'  # parameter used to perform distributed training
        self.nb_iterations = 370000  # total number of iterations to perform during training
        self.iters_per_checkpoint = 10000  # number of iterations between successive checkpoints
        self.iters_check_for_model_improvement = 5000  # number of iterations between successive evaluation on the validation set
        self.batch_size = 16  # batch size per GPU card
        self.accumulation_steps = 3  # number of iterations before updating model parameters (gradient accumulation)
        self.checkpoint = ''  # checkpoint to use to restart training at a specific place
        
        # loss weigths hyper-parameters
        self.lambda_reversal = 1.  # lambda multiplier used in reversal gradient layer
        self.adv_max_weight = 1e-2  # max weight to apply on speaker adversarial loss
        self.post_mult_weight = 1e-3  # weight to apply on FiLM scalar post-multipliers
        self.dur_weight = 1.  # weight to apply on duration loss
        self.energy_weight = 1.  # weight to apply on energy loss
        self.pitch_weight = 1.  # weight to apply on pitch loss
        self.mel_spec_weight = 1.  # weight to apply on mel-spec loss
        
        # optimizer hyper-parameters
        self.optimizer = 'adam'  # optimizer to use for training
        self.betas = (0.9, 0.98)  # betas coefficients in Adam
        self.epsilon = 1e-9  # used for numerical stability in Adam
        self.weight_decay = 1e-6  # weight decay (L2 regularization) to use in the optimizer
        self.initial_learning_rate = 1e-4  # value of learning rate at iteration 0
        self.max_learning_rate = 1e-3  # max value of learning rate during training
        self.warmup_steps = 10000  # linearly increase the learning rate for the first warmup steps
        self.grad_clip_thresh = float('inf')  # gradient clipping threshold to stabilize training
        
        # Daft-Exprt module hyper-parameters
        self.prosody_encoder = {
            'nb_blocks': 4,
            'hidden_embed_dim': 128,
            'attn_nb_heads': 8,
            'attn_dropout': 0.1,
            'conv_kernel': 3,
            'conv_channels': 1024,
            'conv_dropout': 0.1
        }
        
        self.phoneme_encoder = {
            'nb_blocks': 4,
            'hidden_embed_dim': 128,
            'attn_nb_heads': 2,
            'attn_dropout': 0.1,
            'conv_kernel': 3,
            'conv_channels': 1024,
            'conv_dropout': 0.1
        }
        
        self.local_prosody_predictor = {
            'nb_blocks': 1,
            'conv_kernel': 3,
            'conv_channels': 256,
            'conv_dropout': 0.1,
        }
        
        self.gaussian_upsampling_module = {
            'conv_kernel': 3
        }
        
        self.frame_decoder = {
            'nb_blocks': 4,
            'attn_nb_heads': 2,
            'attn_dropout': 0.1,
            'conv_kernel': 3,
            'conv_channels': 1024,
            'conv_dropout': 0.1
        }
        
        ######################################################################
        #### hyper-parameter values that have to be specified in **kwargs ####
        ######################################################################
        self.training_files = None  # path to training files
        self.validation_files = None  # path to validation files
        self.output_directory = None  # path to save training outputs (checkpoints, config files, audios, logging ...)

        self.language = None  # spoken language of the speaker(s)
        self.speakers = None  # speakers we want to use for training or transfer learning
        
        ##########################################################################################
        #### hyper-parameter inferred from other hyper-params values or specified in **kwargs ####
        ##########################################################################################
        self.stats = {}  # features stats used during training and inference
        self.symbols = []  # list of symbols used in the specified language
        
        self.n_speakers = 0  # number of speakers to use with a lookup table
        self.speakers_id = []  # ID associated to each speaker -- starts from 0

        ########################################################
        #### update hyper-parameter variables with **kwargs ####
        ########################################################
        for key, value in kwargs.items():
            if hasattr(self, key) and getattr(self, key) is not None and getattr(self, key) != value and verbose:
                _logger.warning(f'Changing parameter "{key}" = {value} (was {getattr(self, key)})')
            setattr(self, key, value)

        # check if all hyper-params have an assigned value
        for param, value in self.__dict__.items():
            assert(value is not None), _logger.error(f'Hyper-parameter "{param}" is None -- please specify a value')

        # give a default value to hyper-parameters that have not been specified in **kwargs
        self._set_default_hyper_params(verbose=verbose)

    def _set_default_hyper_params(self, verbose):
        ''' Give a default value to hyper-parameters that have not been specified in **kwargs

        :param verbose:         whether to display logger info/warnings or not
        '''
        # update MFA paths
        self.update_mfa_paths()
        # set stats if not already set
        stats_file = os.path.join(self.output_directory, 'stats.json')
        if len(self.stats) == 0 and os.path.isfile(stats_file):
            with open(stats_file) as f:
                data = f.read()
            stats = json.loads(data)
            self.stats = stats

        # set symbols if not already set
        if len(self.symbols) == 0:
            if self.language == 'english':
                self.symbols = symbols_english
            else:
                _logger.error(f'Language: {self.language} -- No default value for "symbols" -- please specify a value')
                sys.exit(1)
            if verbose:
                _logger.info(f'Language: {self.language} -- {len(self.symbols)} symbols used')
        # set number of symbols
        self.n_symbols = len(self.symbols)
        # check padding symbol is at index 0
        # zero padding is used in the DataLoader and Daft-Exprt model
        assert(self.symbols.index(pad) == 0), _logger.error(f'Padding symbol "{pad}" must be at index 0')
        
        # set speakers ID if not already set
        if len(self.speakers_id) == 0:
            self.speakers_id = [i for i in range(len(self.speakers))]
            if verbose:
                _logger.info(f'Nb speakers: {len(self.speakers)} -- Changed "speakers_id" to {self.speakers_id}')
        # set n_speakers if not already set
        if self.n_speakers == 0:
            self.n_speakers = len(set(self.speakers_id)) + 1
            if verbose:
                _logger.info(f'Nb speakers: {len(set(self.speakers_id))} -- Changed "n_speakers" to {self.n_speakers}\n')
        
        # check number of speakers is coherent
        assert (self.n_speakers >= len(set(self.speakers_id))), \
            _logger.error(f'Parameter "n_speakers" must be superior or equal to the number of speakers -- '
                          f'"n_speakers" = {self.n_speakers} -- Number of speakers = {len(set(self.speakers_id))}')
        # check items in the lists are unique and have the same size
        assert (len(self.speakers) == len(set(self.speakers))), \
            _logger.error(f'Speakers are not unique: {len(self.speakers)} -- {len(set(self.speakers))}')
        assert (len(self.speakers) == len(self.speakers_id)), \
            _logger.error(f'Parameters "speakers" and "speakers_id" don\'t have the same length: '
                          f'{len(self.speakers)} -- {len(self.speakers_id)}')

        # check FFT/Mel-Spec extraction parameters are correct
        assert(self.filter_length % self.hop_length == 0), _logger.error(f'filter_length must be a multiple of hop_length')
    
    def update_mfa_paths(self):
        ''' Update MFA paths to match the ones in the current environment
        '''
        # paths used by MFA
        home = str(Path.home())
        self.mfa_dictionary = os.path.join(home, 'Documents', 'MFA', 'pretrained_models', 'dictionary', f'{self.language}.dict')
        self.mfa_g2p_model = os.path.join(home, 'Documents', 'MFA', 'pretrained_models', 'g2p', f'{self.language}_g2p.zip')
        self.mfa_acoustic_model = os.path.join(home, 'Documents', 'MFA', 'pretrained_models', 'acoustic', f'{self.language}.zip')
        # check MFA files exist
        assert(os.path.isfile(self.mfa_dictionary)), _logger.error(f'There is no such file "{self.mfa_dictionary}"')
        assert(os.path.isfile(self.mfa_g2p_model)), _logger.error(f'There is no such file "{self.mfa_g2p_model}"')
        assert(os.path.isfile(self.mfa_acoustic_model)), _logger.error(f'There is no such file "{self.mfa_acoustic_model}"')
    
    def save_hyper_params(self, json_file):
        ''' Save hyper-parameters to JSON file
        
        :param json_file:       path of the JSON file to store hyper-parameters
        '''
        # create directory if it does not exists
        dirname = os.path.dirname(json_file)
        os.makedirs(dirname, exist_ok=True)
        # extract hyper-parameters used
        hyper_params = self.__dict__.copy() 
        # save hyper-parameters to JSON file
        with open(json_file, 'w') as f:
            json.dump(hyper_params, f, indent=4, sort_keys=True)
