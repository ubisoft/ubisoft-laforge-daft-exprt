import random

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from daft_exprt.extract_features import duration_to_integer
from daft_exprt.utils import histogram_plot, plot_2d_data, scatter_plot


class DaftExprtLogger(SummaryWriter):
    def __init__(self, logdir):
        super(DaftExprtLogger, self).__init__(logdir)

    def log_training(self, loss, indiv_loss, grad_norm, learning_rate, duration, iteration):
        ''' Log training info

        :param loss:                training batch loss
        :param indiv_loss:          training batch individual losses
        :param grad_norm:           norm of the gradient
        :param learning_rate:       learning rate
        :param duration:            duration per iteration
        :param iteration:           current training iteration
        '''
        self.add_scalars("DaftExprt.optimization", {'grad_norm': grad_norm, 'learning_rate': learning_rate,
                                                     'duration': duration}, iteration)
        self.add_scalars("DaftExprt.training", {'training_loss': loss}, iteration)
        for key in indiv_loss:
            if indiv_loss[key] != 0:
                if 'loss' in key:
                    self.add_scalars(f"DaftExprt.training", {f'{key}': indiv_loss[key]}, iteration)

    def log_validation(self, val_loss, val_indiv_loss, val_targets, val_outputs, model, hparams, iteration):
        ''' Log validation info

        :param val_loss:                validation loss
        :param val_indiv_loss:          individual validation losses
        :param val_targets:             list of ground-truth values on the valid set
        :param val_outputs:             list of predicted values on the valid set
        :param model:                   model used for training/validation
        :param hparams:                 hyper-parameters used for training
        :param iteration:               current training iteration
        '''
        # plot validation losses
        self.add_scalars("DaftExprt.validation", {"validation_loss": val_loss}, iteration)
        for key in val_indiv_loss:
            self.add_scalars("DaftExprt.validation", {f'{key}': val_indiv_loss[key]}, iteration)

        # # plot distribution of model parameters
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     self.add_histogram(tag, value.detach().cpu().numpy(), iteration)

        # choose random index to extract batch of targets and outputs
        idx = random.randint(0, len(val_targets) - 1)
        targets = val_targets[idx]
        outputs = val_outputs[idx]
        # extract predicted outputs and ground-truth values
        duration_targets, energy_targets, pitch_targets, mel_spec_targets, _ = targets
        _, _, encoder_preds, decoder_preds, alignments = outputs
        duration_preds, energy_preds, pitch_preds, input_lengths = encoder_preds
        mel_spec_preds, output_lengths = decoder_preds
        weights = alignments
        # choose random index in the batch
        idx = random.randint(0, mel_spec_preds.size(0) - 1)
        # extract corresponding sequence length
        input_length = input_lengths[idx].item()
        output_length = output_lengths[idx].item()
        # transfer data to cpu and convert to numpy array
        duration_targets = duration_targets[idx, :input_length].detach().cpu().numpy()  # (L, )
        duration_preds = duration_preds[idx, :input_length].detach().cpu().numpy()  # (L, )
        energy_targets = energy_targets[idx, :input_length].detach().cpu().numpy()  # (L, )
        energy_preds = energy_preds[idx, :input_length].detach().cpu().numpy()  # (L, )
        pitch_targets = pitch_targets[idx, :input_length].detach().cpu().numpy()  # (L, )
        pitch_preds = pitch_preds[idx, :input_length].detach().cpu().numpy()  # (L, )
        mel_spec_targets = mel_spec_targets[idx, :, :output_length].detach().cpu().numpy()  # (n_mel_channels, T)
        mel_spec_preds = mel_spec_preds[idx, :, :output_length].detach().cpu().numpy()  # (n_mel_channels, T)
        weights = weights[idx, :input_length, :output_length].detach().cpu().numpy()  # (L, T)
        
        # convert target float durations to int durations
        duration_int_targets = np.zeros(len(duration_targets), dtype='int32')  # (L, )
        end_prev, symbols_idx, durations_float = 0., [], []
        for symbol_id in range(len(duration_targets)):
            symb_dur = duration_targets[symbol_id]
            if symb_dur != 0.:  # ignore 0 durations
                symbols_idx.append(symbol_id)
                durations_float.append([end_prev, end_prev + symb_dur])
                end_prev += symb_dur
        int_durs = duration_to_integer(durations_float, hparams)  # (L, )
        duration_int_targets[symbols_idx] = int_durs  # (L, )
        # extract target alignments
        col_idx = 0
        alignment_targets = np.zeros((len(duration_int_targets), mel_spec_targets.shape[1]))  # (L, T)
        for symbol_id in range(alignment_targets.shape[0]):
            nb_frames = duration_int_targets[symbol_id]
            alignment_targets[symbol_id, col_idx: col_idx + nb_frames] = 1.
            col_idx += nb_frames
        
        # extract all FiLM parameter predictions on the validation set
        # FiLM parameters for Encoder Module
        encoder_film = [output[1][1] for output in val_outputs]  # (B, nb_blocks, nb_film_params)
        encoder_film = torch.cat(encoder_film, dim=0)  # (B_tot, nb_blocks, nb_film_params)
        encoder_film = encoder_film.detach().cpu().numpy()  # (B_tot, nb_blocks, nb_film_params)
        # FiLM parameters for Prosody Predictor Module
        prosody_pred_film = [output[1][2] for output in val_outputs] # (B, nb_blocks, nb_film_params)
        prosody_pred_film = torch.cat(prosody_pred_film, dim=0)  # (B_tot, nb_blocks, nb_film_params)
        prosody_pred_film = prosody_pred_film.detach().cpu().numpy()  # (B_tot, nb_blocks, nb_film_params)
        # FiLM parameters for Decoder Module
        decoder_film = [output[1][3] for output in val_outputs] # (B, nb_blocks, nb_film_params)
        decoder_film = torch.cat(decoder_film, dim=0)  # (B_tot, nb_blocks, nb_film_params)
        decoder_film = decoder_film.detach().cpu().numpy()  # (B_tot, nb_blocks, nb_film_params)
        
        # plot histograms of gammas and betas for each block of each module
        for module, tensor in zip(['encoder', 'prosody_predictor', 'decoder'],
                                  [encoder_film, prosody_pred_film, decoder_film]):
            nb_blocks = tensor.shape[1]
            nb_gammas = int(tensor.shape[2] / 2)
            gammas = histogram_plot(data=[tensor[:, i, :nb_gammas] for i in range(nb_blocks)],
                                    x_labels=[f'Value -- Block {i}' for i in range(nb_blocks)],
                                    y_labels=['Frequency' for _ in range(nb_blocks)])
            self.add_figure(tag=f'{module} -- FiLM gammas', figure=gammas, global_step=iteration)
            betas = histogram_plot(data=[tensor[:, i, nb_gammas:] for i in range(nb_blocks)],
                                   x_labels=[f'Value -- Block {i}' for i in range(nb_blocks)],
                                   y_labels=['Frequency' for _ in range(nb_blocks)])
            self.add_figure(tag=f'{module} -- FiLM betas', figure=betas, global_step=iteration)
        # plot duration target and duration pred
        durations = scatter_plot(data=(duration_targets, duration_preds),
                                 colors=('blue', 'red'),
                                 labels=('ground-truth', 'predicted'),
                                 x_label='Symbol ID',
                                 y_label='Duration (sec)')
        self.add_figure(tag='durations', figure=durations, global_step=iteration)
        # plot energy target and energy pred
        energies = scatter_plot(data=(energy_targets, energy_preds),
                                colors=('blue', 'red'),
                                labels=('ground-truth', 'predicted'),
                                x_label='Symbol ID',
                                y_label='Energy (normalized)')
        self.add_figure(tag='energies', figure=energies, global_step=iteration)
        # plot pitch target and pitch pred
        pitch = scatter_plot(data=(pitch_targets, pitch_preds),
                             colors=('blue', 'red'),
                             labels=('ground-truth', 'predicted'),
                             x_label='Symbol ID',
                             y_label='Pitch (normalized)')
        self.add_figure(tag='pitch', figure=pitch, global_step=iteration)
        # plot mel-spec target and mel-spec pred
        mel_specs = plot_2d_data(data=(mel_spec_targets, mel_spec_preds),
                                 x_labels=('Frames -- Ground Truth', 'Frames -- Predicted'),
                                 y_labels=('Frequencies', 'Frequencies'))
        self.add_figure(tag='mel-spectrogram', figure=mel_specs, global_step=iteration)
        # plot alignment target and alignment pred
        alignments = plot_2d_data(data=(alignment_targets, weights),
                                  x_labels=('Frames -- Ground Truth', 'Frames -- Predicted (from Ground Truth)'),
                                  y_labels=('Symbol ID', 'Symbol ID'))
        self.add_figure(tag='alignments', figure=alignments, global_step=iteration)
