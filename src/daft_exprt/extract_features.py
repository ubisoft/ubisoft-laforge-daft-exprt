import json
import logging
import logging.handlers
import os
import re
import subprocess
import types
import uuid

import librosa
import numpy as np
import torch

from shutil import rmtree

from librosa.filters import mel as librosa_mel_fn
from scipy.io import wavfile

from daft_exprt.symbols import ascii, eos, punctuation, SIL_WORD_SYMBOL, whitespace
from daft_exprt.utils import launch_multi_process


_logger = logging.getLogger(__name__)
FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
TMP_DIR = os.path.join(FILE_ROOT, 'tmp')
FEATURES_HPARAMS = ['centered', 'cutoff', 'f0_interval', 'filter_length', 'hop_length',
                    'language', 'mel_fmax', 'mel_fmin', 'min_clipping', 'max_f0', 'min_f0',
                    'n_mel_channels', 'order', 'sampling_rate', 'symbols', 'uv_cost', 'uv_interval']


def check_features_config_used(features_dir, hparams):
    ''' Check current config is the same than the one used in features directory
    '''
    # hyper-params that are important for feature extraction
    same_config = True
    for root, _, file_names in os.walk(os.path.normpath(features_dir)):
        # extract config files
        configs = [x for x in file_names if x.endswith('.json')]
        if len(configs) != 0:
            # get previous config
            with open(os.path.join(root, configs[0])) as f:
                data = f.read()
            config = json.loads(data)
            hparams_prev = types.SimpleNamespace(**config)
            # compare params
            for param in FEATURES_HPARAMS:
                if getattr(hparams, param) != getattr(hparams_prev, param):
                    same_config = False
                    _logger.warning(f'Parameter "{param}" is different in "{root}" -- '
                                    f'Was {getattr(hparams_prev, param)} and now is {getattr(hparams, param)}')
    
    return same_config


def get_min_phone_duration(lines, min_phone_dur=1000.):
    ''' Extract shortest phone duration in the current .markers file
    '''
    # iterate over phones
    for line in lines:
        line = line.strip().split(sep='\t')
        # extract phone duration
        begin, end = float(line[0]), float(line[1])
        if end - begin < min_phone_dur:
            min_phone_dur = end - begin
    
    return min_phone_dur


def duration_to_integer(float_durations, hparams, nb_samples=None):
    ''' Convert phoneme float durations to integer frame durations
    '''
    # estimate number of samples in audio
    if nb_samples is None:
        # get total duration of audio
        # float_durations = [[phone_begin, phone_end], ...]
        total_duration = sum([(x[1] - x[0]) for x in float_durations])
        # convert in number of samples
        nb_samples = int(total_duration * hparams.sampling_rate)
    # get nb spectrogram frames
    # ignore padding for the moment
    nb_frames = 1 + int((nb_samples - hparams.filter_length) / hparams.hop_length)
    # get spectrogram frames index
    frames_idx = [int(hparams.filter_length / 2) + hparams.hop_length * i for i in range(nb_frames)]
    
    # compute number of frames per phoneme
    curr_frame = 1
    int_durations = []
    while curr_frame <= nb_frames:
        # extract phoneme duration
        begin, end = float_durations.pop(0)
        if begin != end:
            # convert to sample idx
            begin, end = int(begin * hparams.sampling_rate), int(end * hparams.sampling_rate)
            # get corresponding frames
            nb_phone_frames = len([idx for idx in frames_idx if begin < idx <= end])
            int_durations.append(nb_phone_frames)
            curr_frame += nb_phone_frames
        else:  # we should not have 0 durations
            raise ValueError
    # add edge frames if padding is on
    if hparams.centered:
        nb_edge_frames = int(hparams.filter_length / 2 / hparams.hop_length)
        # left padding
        int_durations[0] += nb_edge_frames
        # right padding
        if len(float_durations) != 0:  # correspond to last phoneme
            int_durations.append(nb_edge_frames)
        else:
            int_durations[-1] += nb_edge_frames
    
    return int_durations


def update_markers(file_name, lines, sentence, sent_begin, int_durations, hparams, logger):
    ''' Update markers:
        - change timings to start from 0
        - add punctuation or whitespace at word boundaries
        - add EOS token at end of sentence
        - add int durations
    '''
    # characters to consider in the sentence
    if hparams.language == 'english':
        all_chars = ascii + punctuation
    else:
        raise NotImplementedError()
    
    '''
        match words in the sentence with the ones in markers lines
        Sentence:         ,THAT's, an example'! ' of a sentence. . .'
        Markers words:    that s an example <sil> of a sentence
    '''
    # split sentence:
    # [',', "that's", ',', 'an', "example'", '!', "'", 'of', 'a', 'sentence', '.', '.', '.', "'"]
    sent_words = re.findall(f"[\w']+|[{punctuation}]", sentence.lower().strip())
    # remove characters that are not letters or punctuation:
    # [',', "that's", ',', 'an', "example'", '!', 'of', 'a', 'sentence', '.', '.', '.']
    sent_words = [x for x in sent_words if len(re.sub(f'[^{all_chars}]', '', x)) != 0]
    # be sure to begin the sentence with a word and not a punctuation
    # ["that's", ',', 'an', "example'", '!', 'of', 'a', 'sentence', '.', '.', '.']
    while sent_words[0] in punctuation:
        sent_words.pop(0)
    # keep only one punctuation type at the end
    # ["that's", ',', 'an', "example'", '!', 'of', 'a', 'sentence']
    punctuation_end = None
    while sent_words[-1] in punctuation:
        punctuation_end = sent_words.pop(-1)
    
    # split markers lines -- [[begin, end, phone, word, word_idx], ....]
    markers = [line.strip().split(sep='\t') for line in lines]
    # extract markers words
    # they are no '<sil>' at beginning and end of sentence because we trimmed the audio
    # ['that', 's', 'an', example'', '<sil>', 'of', 'a', 'sentence']
    words_idx = [marker[4] for marker in markers]
    lines_idx = [words_idx.index(word_idx) for word_idx in list(dict.fromkeys(words_idx).keys())]
    marker_words = [markers[line_idx][3] for line_idx in lines_idx]
    
    # update markers with word boundaries
    sent_words_copy, markers_old = sent_words.copy(), markers.copy()
    markers, word_idx, word_error = [], 0, False
    while len(sent_words) != 0:
        # extract word in .lab sentence and .markers file
        sent_word = sent_words.pop(0)
        marker_word, marker_word_idx = markers_old[0][3], markers_old[0][4]
        if marker_word != sent_word:
            # we should have the same words
            # generally the issue comes from the symbol '
            # e.g. example' vs example or that's vs [that, s]
            regex_word = re.findall(f"[\w]+|[{punctuation}]", sent_word)
            if len(regex_word) == 1:  #  ['example']
                sent_word = regex_word[0]
            else:  #  ['that', 's']
                sent_words = regex_word + sent_words
                sent_word = sent_words.pop(0)
            if marker_word != sent_word:
                # cannot fix the mismatch between words
                word_error = True
                logger.warning(f'Correspondance issue between words in the .lab sentence and those in .markers file -- '
                               f'File name: {file_name} -- Sentence: {sent_words_copy} -- '
                               f'Markers: {marker_words} -- Problematic words: {sent_word} -- {marker_word}')
                break
        # retrieve all markers lines that correspond to the word
        while len(markers_old) != 0 and markers_old[0][4] == marker_word_idx:
            begin, end, phone, word, _ = markers_old.pop(0)
            begin = f'{float(begin) - sent_begin:.3f}'
            end = f'{float(end) - sent_begin:.3f}'
            int_dur = str(int_durations.pop(0))
            markers.append([begin, end, int_dur, phone, word, str(word_idx)])
        # at this point we pass to the next word
        # we must add a word boundary between two consecutive words
        word_idx += 1
        if len(sent_words) != 0:
            word_bound = sent_words.pop(0) if sent_words[0] in punctuation else whitespace
            # check if a silence marker is associated to the word boundary
            if markers_old[0][3] == SIL_WORD_SYMBOL:
                begin, end, _, _, _ = markers_old.pop(0)
                begin = f'{float(begin) - sent_begin:.3f}'
                end = f'{float(end) - sent_begin:.3f}'
                int_dur = str(int_durations.pop(0))
                markers.append([begin, end, int_dur, word_bound, word_bound, str(word_idx)])
            else:
                end_prev = markers[-1][1]
                markers.append([end_prev, end_prev, str(0), word_bound, word_bound, str(word_idx)])
            word_idx += 1
    
    if not word_error:
        # add end punctuation if there is one
        if punctuation_end is not None:
            end_prev = markers[-1][1]
            markers.append([end_prev, end_prev, str(0), punctuation_end, punctuation_end, str(word_idx)])
            word_idx += 1
        # add EOS token
        end_prev = markers[-1][1]
        markers.append([end_prev, end_prev, str(0), eos, eos, str(word_idx)])
        # check everything is correct
        assert(len(sent_words) == len(markers_old) == len(int_durations) == 0), \
            logger.error(f'File name: {file_name} -- length mismatch between lists: ({sent_words}, {markers_old}, {int_durations})')
        return markers
    else:
        return None


def extract_pitch(wav, fs, hparams):
    ''' Extract pitch frames from audio using REAPER binary
        Convert pitch to log scale and set unvoiced values to 0.
    '''
    # REAPER asks for int16 audios
    # audio is in float32
    wav = wav * 32768.0
    wav = wav.astype('int16')
    # save audio file locally
    rand_name = str(uuid.uuid4())
    out_dir = os.path.join(TMP_DIR, 'reaper')
    os.makedirs(out_dir, exist_ok=True)
    wav_file = os.path.join(out_dir, f'{rand_name}.wav')
    wavfile.write(wav_file, fs, wav)
    
    # extract pitch values
    f0_file = wav_file.replace('.wav', '.f0')
    process = ['reaper', '-i', f'{wav_file}',
               '-a', '-f', f'{f0_file}',
               '-e', f'{hparams.f0_interval}',
               '-m', f'{hparams.min_f0}',
               '-x', f'{hparams.max_f0}',
               '-u', f'{hparams.uv_interval}',
               '-w', f'{hparams.uv_cost}']
    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call(process, stdout=devnull, stderr=subprocess.STDOUT)
    # read PCM file
    with open(f0_file, 'rb') as f:
        buf = f.read()
        pitch = np.frombuffer(buf, dtype='int16')
    # extract unvoiced indexes
    pitch = np.copy(pitch)
    uv_idxs = np.where(pitch <= 0.)[0]
    # put to log scale
    pitch[uv_idxs] = 1000.
    pitch = np.log(pitch)
    # set unvoiced values to 0.
    pitch[uv_idxs] = 0.
    # extract pitch for each mel-spec frame
    pitch_frames = pitch[::hparams.hop_length]
    # edge case
    if len(pitch) % hparams.hop_length == 0:
        pitch_frames = np.append(pitch_frames, pitch[-1])
    # delete files
    os.remove(wav_file)
    os.remove(f0_file)
    
    return pitch_frames


def get_symbols_pitch(pitch, markers):
    ''' Compute mean pitch per symbol

        pitch = NumPy array of shape (nb_mel_spec_frames, )
        markers = [[begin, end, int_dur, symbol, word, word_idx], ...]
    '''
    idx = 0
    symbols_pitch = []
    for marker in markers:
        # number of mel-spec frames assigned to the symbol
        int_dur = int(marker[2])
        if int_dur != 0:
            # ignore unvoiced values
            symbol_pitch = pitch[idx: idx + int_dur]
            symbol_pitch = symbol_pitch[symbol_pitch > 0.]
            # compute mean pitch for voiced values
            if len(symbol_pitch) != 0:
                symbols_pitch.append(f'{np.mean(symbol_pitch):.3f}\n')
            else:
                symbols_pitch.append(f'{0.:.3f}\n')
            idx += int_dur
        else:
            symbols_pitch.append(f'{0.:.3f}\n')
    
    return symbols_pitch


def extract_energy(mel_spec):
    ''' Extract energy of each mel-spec frame
        mel_spec = NumPy array of shape (nb_mel_spec_channels, nb_mel_spec_frames)
    '''
    energy = np.linalg.norm(mel_spec, axis=0)
    return energy


def get_symbols_energy(energy, markers):
    ''' Compute mean energy per symbol

        energy = NumPy array of shape (nb_mel_spec_frames, )
        markers = [[begin, end, int_dur, symbol, word, word_idx], ...]
    '''
    idx = 0
    symbols_energy = []
    for marker in markers:
        # number of mel-spec frames assigned to the symbol
        int_dur = int(marker[2])
        if int_dur != 0:
            # compute mean energy
            symbol_energy = energy[idx: idx + int_dur]
            symbol_energy = np.mean(symbol_energy)
            symbols_energy.append(f'{symbol_energy:.3f}\n')
            idx += int_dur
        else:
            symbols_energy.append(f'{0.:.3f}\n')
    
    return symbols_energy


def mel_spectrogram_HiFi(wav, hparams):
    ''' Mel-Spectrogram extraction as it is performed by HiFi-GAN
    '''
    # convert to PyTorch float tensor
    wav = torch.FloatTensor(wav)  # (T, )
    # extract hparams
    fmin = hparams.mel_fmin
    fmax = hparams.mel_fmax
    center = hparams.centered
    hop_size = hparams.hop_length
    n_fft = hparams.filter_length
    num_mels = hparams.n_mel_channels
    sampling_rate = hparams.sampling_rate
    min_clipping = hparams.min_clipping
    # get mel filter bank
    mel_filter_bank = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)  # (n_mels, 1 + n_fft/2)
    mel_filter_bank = torch.from_numpy(mel_filter_bank).float()  # (n_mels, 1 + n_fft/2)
    # build hann window
    hann_window = torch.hann_window(n_fft)
    # extract amplitude spectrogram
    spec = torch.stft(wav, n_fft, hop_length=hop_size, win_length=n_fft, window=hann_window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    # convert to mels and pass to log
    mel_spec = torch.matmul(mel_filter_bank, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=min_clipping))
    # transform to numpy array
    mel_spec = mel_spec.squeeze().numpy()

    return mel_spec


def rescale_wav_to_float32(x):
    ''' Rescale audio array between -1.f and 1.f based on the current format
    '''
    # convert
    if x.dtype == 'int16':
        y = x / 32768.0
    elif x.dtype == 'int32':
        y = x / 2147483648.0
    elif x.dtype == 'uint8':
        y = ((x / 255.0) - 0.5)*2
    elif x.dtype == 'float32' or x.dtype == 'float64':
        y = x
    else:
        raise TypeError(f"could not normalize wav, unsupported sample type {x.dtype}")
    # check amplitude is correct
    y = y.astype('float32')
    max_ampl = np.max(np.abs(y))
    if max_ampl > 1.0:
        pass  # the error should be raised but librosa returns values bigger than 1 sometimes
        # raise ValueError(f'float32 wav contains samples not in the range [-1., 1.] -- '
        #                  f'max amplitude: {max_ampl}')

    return y


def _extract_features(files, features_dir, hparams, log_queue):
    ''' Extract mel-spectrogram and markers with int duration
    '''
    # create logger from logging queue
    qh = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    if not root.hasHandlers():
        root.setLevel(logging.INFO)
        root.addHandler(qh)
    logger = logging.getLogger(f"worker{str(uuid.uuid4())}")
    
    # check files exist
    markers_file, wav_file = files
    assert(os.path.isfile(markers_file)), logger.error(f'There is no such file: {markers_file}')
    assert(os.path.isfile(wav_file)), logger.error(f'There is no such file: {wav_file}')
    # read markers lines
    with open(markers_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # check min phone duration is coherent
    # min phone duration must be >= filter_length // 2 
    # in order to have at least one mel-spec frame attributed to the phone
    min_phone_dur = get_min_phone_duration(lines)
    fft_length = hparams.filter_length / hparams.sampling_rate
    assert(min_phone_dur > fft_length / 2), \
        logger.error(f'Min phone duration = {min_phone_dur} -- filter_length / 2 = {fft_length / 2}')
    
    # extract sentence duration
    # leading and tailing silences have been removed in markers.py script
    sent_begin = float(lines[0].strip().split(sep='\t')[0])
    sent_end = float(lines[-1].strip().split(sep='\t')[1])
    sent_dur = sent_end - sent_begin
    
    # ignore audio if length is inferior to min wav duration
    if sent_dur >= hparams.minimum_wav_duration / 1000:
        # read wav file to range [-1, 1] in np.float32
        wav, fs = librosa.load(wav_file, sr=hparams.sampling_rate)
        wav = rescale_wav_to_float32(wav)
        # remove leading and tailing silences
        wav = wav[int(sent_begin * fs): int(sent_end * fs)]
        
        # extract mel-spectrogram
        mel_spec = mel_spectrogram_HiFi(wav, hparams)
        # get number of mel-spec frames
        nb_mel_spec_frames = mel_spec.shape[1]
        
        # convert phoneme durations to integer frame durations
        float_durations = [[float(x[0]) - sent_begin, float(x[1]) - sent_begin]
                           for x in [line.strip().split(sep='\t') for line in lines]]
        int_durations = duration_to_integer(float_durations, hparams, nb_samples=len(wav))
        assert(len(int_durations) == len(lines)), logger.error(f'{markers_file} -- ({len(int_durations)}, {len(lines)})')
        assert(sum(int_durations) == nb_mel_spec_frames), logger.error(f'{markers_file} -- ({sum(int_durations)}, {nb_mel_spec_frames})')
        assert(0 not in int_durations), logger.error(f'{markers_file} -- {int_durations}')
        
        # update markers:
        # change timings to start from 0
        # add punctuation or whitespace at word boundaries
        # add EOS token at end of sentence
        # add int durations
        markers_dir = os.path.dirname(markers_file)
        file_name = os.path.basename(markers_file).replace('.markers', '')
        sentence_file = os.path.join(markers_dir, f'{file_name}.lab')
        assert(os.path.isfile(sentence_file)), logger.error(f'There is no such file: {sentence_file}')
        with open(sentence_file, 'r', encoding='utf-8') as f:
            sentence = f.readline()
        markers = update_markers(file_name, lines, sentence, sent_begin, int_durations, hparams, logger)
        
        if markers is not None:
            # save mel-spectrogram -- (n_mel_channels, T)
            np.save(os.path.join(features_dir, f'{file_name}.npy'), mel_spec)
            
            # save markers
            # each line has the format: [begin, end, int_dur, symbol, word, word_idx]
            markers_file = os.path.join(features_dir, f'{file_name}.markers')
            with open(markers_file, 'w', encoding='utf-8') as f:
                f.writelines(['\t'.join(x) + '\n' for x in markers])
            
            # extract energy for each mel-spec frame
            mel_spec = np.exp(mel_spec)  # remove log
            frames_energy = extract_energy(mel_spec)
            # save frames energy values
            energy_file = os.path.join(features_dir, f'{file_name}.frames_nrg')
            with open(energy_file, 'w', encoding='utf-8') as f:
                for val in frames_energy:
                    f.write(f'{val:.3f}\n')
            # extract energy on the symbol level
            # we use average energy value per symbol
            symbols_energy = get_symbols_energy(frames_energy, markers)
            # save symbols energy
            energy_file = os.path.join(features_dir, f'{file_name}.symbols_nrg')
            with open(energy_file, 'w', encoding='utf-8') as f:
                f.writelines(symbols_energy)
            
            # extract log pitch for each mel-spec frame
            frames_pitch = extract_pitch(wav, fs, hparams)
            assert(len(frames_pitch) == nb_mel_spec_frames), logger.error(f'{markers_file} -- ({len(frames_pitch)}, {nb_mel_spec_frames})')
            # save frames pitch values
            pitch_file = os.path.join(features_dir, f'{file_name}.frames_f0')
            with open(pitch_file, 'w', encoding='utf-8') as f:
                for val in frames_pitch:
                    f.write(f'{val:.3f}\n')
            # extract pitch on the symbol level
            # we use average pitch value per symbol
            symbols_pitch = get_symbols_pitch(frames_pitch, markers)
            # save symbols pitch values
            pitch_file = os.path.join(features_dir, f'{file_name}.symbols_f0')
            with open(pitch_file, 'w', encoding='utf-8') as f:
                f.writelines(symbols_pitch)
    else:
        logger.warning(f'Ignoring {wav_file} -- audio has length inferior to {hparams.minimum_wav_duration / 1000}s after trimming')


def get_files_for_features_extraction(line, markers_dir, log_queue):
    ''' Return file name if .markers file exists
    '''
    # check if markers file exist for the corresponding line
    line = line.strip().split(sep='|')  # [file_name, text]
    file_name = line[0].strip()
    markers = os.path.join(markers_dir, f'{file_name}.markers')
    if os.path.isfile(markers):
        return file_name
    else:
        return None


def extract_features(dataset_dir, features_dir, hparams, n_jobs):
    ''' Extract features for training
    '''
    # iterate over speakers
    _logger.info('--' * 30)
    _logger.info('Extracting Features'.upper())
    _logger.info('--' * 30)
    for speaker in hparams.speakers:
        _logger.info(f'Speaker: "{speaker}"')
        # check wavs and markers dir exist
        wavs_dir = os.path.join(dataset_dir, speaker, 'wavs')
        markers_dir = os.path.join(dataset_dir, speaker, 'align')
        assert(os.path.isdir(wavs_dir)), _logger.error(f'There is no such directory: {wavs_dir}')
        assert(os.path.isdir(markers_dir)), _logger.error(f'There is no such directory: {markers_dir}')
        # check metadata file exist
        spk_features_dir = os.path.join(features_dir, speaker)
        metadata = os.path.join(spk_features_dir, 'metadata.csv')
        assert(os.path.isfile(metadata)), _logger.error(f'There is no such file: {metadata}')
        
        # get all files that can be used for features extraction
        with open(metadata, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        file_names = launch_multi_process(iterable=lines, func=get_files_for_features_extraction,
                                          n_jobs=n_jobs, markers_dir=markers_dir, timer_verbose=False)
        file_names = [x for x in file_names if x is not None]
        
        # check current files that exist in features dir
        # avoid to process files that already have been processed in a previous features extraction
        curr_files = [x.replace('.symbols_f0', '').strip() for x in os.listdir(spk_features_dir) if x.endswith('.symbols_f0')]
        missing_files = [x for x in file_names if x not in curr_files]
        _logger.info(f'{len(curr_files)} files already processed. {len(missing_files)} new files need to be processed')
        
        # extract features
        files = [(os.path.join(markers_dir, f'{x}.markers'), os.path.join(wavs_dir, f'{x}.wav')) for x in missing_files]
        launch_multi_process(iterable=files, func=_extract_features, n_jobs=n_jobs,
                             features_dir=spk_features_dir, hparams=hparams)
        
        # save config used to perform features extraction
        hparams.save_hyper_params(os.path.join(spk_features_dir, 'config.json'))
        _logger.info('')
    # remove tmp directory
    rmtree(TMP_DIR, ignore_errors=True)
