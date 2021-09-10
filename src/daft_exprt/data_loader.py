import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class DaftExprtDataLoader(Dataset):
    ''' Load PyTorch Data Set
        1) load features, symbols and speaker ID
        2) convert symbols to sequence of one-hot vectors
    '''
    def __init__(self, data_file, hparams, shuffle=True):
        # check data file exists and extract lines
        assert(os.path.isfile(data_file))
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.data = [line.strip().split(sep='|') for line in lines]
        self.hparams = hparams
        
        # shuffle
        if shuffle:
            random.seed(hparams.seed)
            random.shuffle(self.data)
    
    def get_mel_spec(self, mel_spec):
        ''' Extract PyTorch float tensor from .npy mel-spec file
        '''
        # transform to PyTorch tensor and check size
        mel_spec = torch.from_numpy(np.load(mel_spec))
        assert(mel_spec.size(0) == self.hparams.n_mel_channels)
        
        return mel_spec
    
    def get_symbols_and_durations(self, markers):
        ''' Extract PyTorch int tensor from an input symbols sequence
            Extract PyTorch float and int duration for each symbol
        '''
        # initialize variables
        symbols, durations_float, durations_int = [], [], []
        
        # read lines of markers file
        with open(markers, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        markers = [line.strip().split(sep='\t') for line in lines]
        
        # iterate over markers
        for marker in markers:
            begin, end, int_dur, symbol, _, _ = marker
            symbols.append(self.hparams.symbols.index(symbol))
            durations_float.append(float(end) - float(begin))
            durations_int.append(int(int_dur))
        
        # convert lists to PyTorch tensors
        symbols = torch.IntTensor(symbols)
        durations_float = torch.FloatTensor(durations_float)
        durations_int = torch.IntTensor(durations_int)
        
        return symbols, durations_float, durations_int
    
    def get_energies(self, energies, speaker_id, normalize=True):
        ''' Extract standardized PyTorch float tensor for energies
        '''
        # read energy lines
        with open(energies, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        energies = np.array([float(line.strip()) for line in lines])
        # standardize energies based on speaker stats
        if normalize:
            zero_idxs = np.where(energies == 0.)[0]
            energies -= self.hparams.stats[f'spk {speaker_id}']['energy']['mean']
            energies /= self.hparams.stats[f'spk {speaker_id}']['energy']['std']
            energies[zero_idxs] = 0.
        # convert to PyTorch float tensor
        energies = torch.FloatTensor(energies)
        
        return energies
    
    def get_pitch(self, pitch, speaker_id, normalize=True):
        ''' Extract standardized PyTorch float tensor for pitch
        '''
        # read pitch lines
        with open(pitch, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        pitch = np.array([float(line.strip()) for line in lines])
        # standardize voiced pitch based on speaker stats
        if normalize:
            zero_idxs = np.where(pitch == 0.)[0]
            pitch -= self.hparams.stats[f'spk {speaker_id}']['pitch']['mean']
            pitch /= self.hparams.stats[f'spk {speaker_id}']['pitch']['std']
            pitch[zero_idxs] = 0.
        # convert to PyTorch float tensor
        pitch = torch.FloatTensor(pitch)
        
        return pitch
    
    def get_data(self, data):
        ''' Extract features, symbols and speaker ID
        '''
        # get mel-spec path, markers path, pitch path and speaker ID
        features_dir = data[0]
        feature_file = data[1]
        speaker_id = int(data[2])
        
        mel_spec = os.path.join(features_dir, f'{feature_file}.npy')
        markers = os.path.join(features_dir, f'{feature_file}.markers')
        symbols_energy = os.path.join(features_dir, f'{feature_file}.symbols_nrg')
        frames_energy = os.path.join(features_dir, f'{feature_file}.frames_nrg')
        symbols_pitch = os.path.join(features_dir, f'{feature_file}.symbols_f0')
        frames_pitch = os.path.join(features_dir, f'{feature_file}.frames_f0')
        
        # extract data
        mel_spec = self.get_mel_spec(mel_spec)
        symbols, durations_float, durations_int = self.get_symbols_and_durations(markers)
        symbols_energy = self.get_energies(symbols_energy, speaker_id)
        frames_energy = self.get_energies(frames_energy, speaker_id, normalize=False)
        symbols_pitch = self.get_pitch(symbols_pitch, speaker_id)
        frames_pitch = self.get_pitch(frames_pitch, speaker_id, normalize=False)
        
        # check everything is correct with sizes
        assert(len(symbols_energy) == len(symbols))
        assert(len(symbols_pitch) == len(symbols))
        assert(len(frames_energy) == mel_spec.size(1))
        assert(len(frames_pitch) == mel_spec.size(1))
        assert(torch.sum(durations_int) == mel_spec.size(1))
        
        return symbols, durations_float, durations_int, symbols_energy, symbols_pitch, \
            frames_energy, frames_pitch, mel_spec, speaker_id, features_dir, feature_file
    
    def __getitem__(self, index):
        return self.get_data(self.data[index])

    def __len__(self):
        return len(self.data)


class DaftExprtDataCollate():
    ''' Zero-pads model inputs and targets
    '''
    def __init__(self, hparams):
        self.hparams = hparams
    
    def __call__(self, batch):
        ''' Collate training batch

        :param batch:   [[symbols, durations_float, durations_int, symbols_energy, symbols_pitch,
                          frames_energy, frames_pitch, mel_spec, speaker_id, features_dir, feature_file], ...]

        :return: collated batch of training samples
        '''
        # find symbols sequence max length
        input_lengths, ids_sorted_decreasing = \
            torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        max_input_len = input_lengths[0]
        
        # right zero-pad sequences to max input length
        symbols = torch.LongTensor(len(batch), max_input_len).zero_()
        durations_float = torch.FloatTensor(len(batch), max_input_len).zero_()
        durations_int = torch.LongTensor(len(batch), max_input_len).zero_()
        symbols_energy = torch.FloatTensor(len(batch), max_input_len).zero_()
        symbols_pitch = torch.FloatTensor(len(batch), max_input_len).zero_()
        speaker_ids = torch.LongTensor(len(batch))
        
        for i in range(len(ids_sorted_decreasing)):
            # extract batch sequences
            symbols_seq = batch[ids_sorted_decreasing[i]][0]
            dur_float_seq = batch[ids_sorted_decreasing[i]][1]
            dur_int_seq = batch[ids_sorted_decreasing[i]][2]
            symbols_energy_seq = batch[ids_sorted_decreasing[i]][3]
            symbols_pitch_seq = batch[ids_sorted_decreasing[i]][4]
            # fill padded arrays
            symbols[i, :symbols_seq.size(0)] = symbols_seq
            durations_float[i, :dur_float_seq.size(0)] = dur_float_seq
            durations_int[i, :dur_int_seq.size(0)] = dur_int_seq
            symbols_energy[i, :symbols_energy_seq.size(0)] = symbols_energy_seq
            symbols_pitch[i, :symbols_pitch_seq.size(0)] = symbols_pitch_seq
            # add corresponding speaker ID
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][8]
        
        # find mel-spec max length
        max_output_len = max([x[7].size(1) for x in batch])
        
        # right zero-pad mel-specs to max output length
        frames_energy = torch.FloatTensor(len(batch), max_output_len).zero_()
        frames_pitch = torch.FloatTensor(len(batch), max_output_len).zero_()
        mel_specs = torch.FloatTensor(len(batch), self.hparams.n_mel_channels, max_output_len).zero_()
        output_lengths = torch.LongTensor(len(batch))
        
        for i in range(len(ids_sorted_decreasing)):
            # extract batch sequences
            frames_energy_seq = batch[ids_sorted_decreasing[i]][5]
            frames_pitch_seq = batch[ids_sorted_decreasing[i]][6]
            mel_spec = batch[ids_sorted_decreasing[i]][7]
            # fill padded arrays
            frames_energy[i, :frames_energy_seq.size(0)] = frames_energy_seq
            frames_pitch[i, :frames_pitch_seq.size(0)] = frames_pitch_seq
            mel_specs[i, :, :mel_spec.size(1)] = mel_spec
            output_lengths[i] = mel_spec.size(1)
        
        # store file identification
        # only used in fine_tune.py script
        feature_dirs, feature_files = [], []
        for i in range(len(ids_sorted_decreasing)):
            feature_dirs.append(batch[ids_sorted_decreasing[i]][9])
            feature_files.append(batch[ids_sorted_decreasing[i]][10])
        
        return symbols, durations_float, durations_int, symbols_energy, symbols_pitch, input_lengths, \
            frames_energy, frames_pitch, mel_specs, output_lengths, speaker_ids, feature_dirs, feature_files


def prepare_data_loaders(hparams, num_workers=1, drop_last=True):
    ''' Initialize train and validation Data Loaders

    :param hparams:             hyper-parameters used for training
    :param num_workers:         number of workers involved in the Data Loader

    :return: Data Loaders for train and validation sets
    '''
    # get data and collate function ready
    train_set = DaftExprtDataLoader(hparams.training_files, hparams)
    val_set = DaftExprtDataLoader(hparams.validation_files, hparams)
    collate_fn = DaftExprtDataCollate(hparams)
    
    # get number of training examples
    nb_training_examples = len(train_set)
    
    # use distributed sampler if we use distributed training
    if hparams.multiprocessing_distributed:
        train_sampler = DistributedSampler(train_set, shuffle=False)
    else:
        train_sampler = None
    
    # build training and validation data loaders
    # drop_last=True because we shuffle data set at each epoch
    train_loader = DataLoader(train_set, num_workers=num_workers, shuffle=(train_sampler is None), sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=True, drop_last=drop_last, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, num_workers=num_workers, shuffle=False, batch_size=hparams.batch_size,
                            pin_memory=True, drop_last=False, collate_fn=collate_fn)
    
    return train_loader, train_sampler, val_loader, nb_training_examples
