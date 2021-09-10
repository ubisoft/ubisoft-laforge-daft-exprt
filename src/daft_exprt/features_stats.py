import collections
import logging
import logging.handlers
import os
import uuid

import numpy as np

from daft_exprt.utils import launch_multi_process


_logger = logging.getLogger(__name__)


def get_symbols_durations(markers_file, hparams, log_queue):
    ''' extract symbols durations in markers file
    '''
    # create logger from logging queue
    qh = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    if not root.hasHandlers():
        root.setLevel(logging.INFO)
        root.addHandler(qh)
    logger = logging.getLogger(f"worker{str(uuid.uuid4())}")
    
    # check file exists
    assert(os.path.isfile(markers_file)), logger.error(f'There is no such file "{markers_file}"')
    # read markers lines
    with open(markers_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    markers = [line.strip().split(sep='\t') for line in lines]  # [[begin, end, nb_frames, symbol, word, word_idx], ...]
    
    # extract duration for each symbol that is in markers
    symbols_durations = []
    for marker in markers:
        begin, end, _, symbol, _, _ = marker
        assert(symbol in hparams.symbols), logger.error(f'{markers_file} -- Symbol "{symbol}" does not exist')
        begin, end = float(begin), float(end)
        symbols_durations.append([symbol, end - begin])
    
    return symbols_durations


def get_non_zero_energy_values(energy_file, log_queue):
    ''' Extract non-zero energy values in energy file
    '''
    # create logger from logging queue
    qh = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    if not root.hasHandlers():
        root.setLevel(logging.INFO)
        root.addHandler(qh)
    logger = logging.getLogger(f"worker{str(uuid.uuid4())}")

    # check file exists
    assert(os.path.isfile(energy_file)), logger.error(f'There is no such file "{energy_file}"')
    # read energy lines
    with open(energy_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    energy_vals = [float(line.strip()) for line in lines]
    # remove non-zero energy values
    energy_vals = list(filter(lambda a: a != 0., energy_vals))
    
    return energy_vals


def get_voiced_pitch_values(pitch_file, log_queue):
    ''' Extract voiced pitch values in pitch file
    '''
    # create logger from logging queue
    qh = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    if not root.hasHandlers():
        root.setLevel(logging.INFO)
        root.addHandler(qh)
    logger = logging.getLogger(f"worker{str(uuid.uuid4())}")

    # check file exists
    assert(os.path.isfile(pitch_file)), logger.error(f'There is no such file "{pitch_file}"')
    # read pitch lines
    with open(pitch_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pitch_vals = [float(line.strip()) for line in lines]
    # remove unvoiced pitch values
    pitch_vals = list(filter(lambda a: a != 0., pitch_vals))
    
    return pitch_vals


def extract_features_stats(hparams, n_jobs):
    ''' Extract features stats for training and inference
    '''
    # only use the training set to extract features stats
    with open(hparams.training_files, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    training_files = [line.strip().split(sep='|') for line in lines]  # [[features_dir, features_file, speaker_id], ...]
    
    # iterate over speakers
    _logger.info('--' * 30)
    _logger.info('Extracting Features Stats'.upper())
    _logger.info('--' * 30)
    symbols_durations = []
    speaker_stats = {f'spk {id}': {'energy': [], 'pitch': []}
                     for id in set(hparams.speakers_id)}
    for speaker_id in set(hparams.speakers_id):
        _logger.info(f'Speaker ID: {speaker_id}')
        # extract all files associated to speaker ID
        spk_training_files = [[x[0], x[1]] for x in training_files if int(x[2]) == speaker_id]
        
        # extract symbol durations
        markers_files = [os.path.join(x[0], f'{x[1]}.markers') for x in spk_training_files]
        symbols_durs = launch_multi_process(iterable=markers_files, func=get_symbols_durations,
                                            n_jobs=n_jobs, hparams=hparams, timer_verbose=False)
        symbols_durs = [y for x in symbols_durs for y in x]
        symbols_durations.extend(symbols_durs)
        
        # extract non-zero energy values
        energy_files = [os.path.join(x[0], f'{x[1]}.symbols_nrg') for x in spk_training_files]
        energy_vals = launch_multi_process(iterable=energy_files, func=get_non_zero_energy_values,
                                           n_jobs=n_jobs, timer_verbose=False)
        energy_vals = [y for x in energy_vals for y in x]
        speaker_stats[f'spk {speaker_id}']['energy'].extend(energy_vals)
        
        # extract voiced symbols pitch values
        pitch_files = [os.path.join(x[0], f'{x[1]}.symbols_f0') for x in spk_training_files]
        pitch_vals = launch_multi_process(iterable=pitch_files, func=get_voiced_pitch_values,
                                          n_jobs=n_jobs, timer_verbose=False)
        pitch_vals = [y for x in pitch_vals for y in x]
        speaker_stats[f'spk {speaker_id}']['pitch'].extend(pitch_vals)
        _logger.info('')
    
    # compute symbols durations stats
    symbols_stats = collections.defaultdict(list)
    for item in symbols_durations:
        symbol, duration = item
        symbols_stats[symbol].append(duration)
    for symbol in symbols_stats:
        min, max = np.min(symbols_stats[symbol]), np.max(symbols_stats[symbol])
        mean, std = np.mean(symbols_stats[symbol]), np.std(symbols_stats[symbol])
        symbols_stats[symbol] = {
            'dur_min': min, 'dur_max': max,
            'dur_mean': mean, 'dur_std': std
        }
    # compute energy and pitch stats for each speaker
    for speaker, vals in speaker_stats.items():
        energy_vals, pitch_vals = vals['energy'], vals['pitch']
        speaker_stats[speaker] = {
            'energy': {
                'mean': np.mean(energy_vals),
                'std': np.std(energy_vals),
                'min': np.min(energy_vals),
                'max': np.max(energy_vals)
            },
            'pitch': {
                'mean': np.mean(pitch_vals),
                'std': np.std(pitch_vals),
                'min': np.min(pitch_vals),
                'max': np.max(pitch_vals)
            }
        }
    # merge stats
    stats = {**speaker_stats}
    stats['symbols'] = symbols_stats
    
    return stats
