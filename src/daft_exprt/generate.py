import collections
import logging
import logging.handlers
import os
import random
import re
import time
import uuid

import librosa
import numpy as np
import torch

from scipy.io import wavfile
from shutil import rmtree

from daft_exprt.cleaners import collapse_whitespace, text_cleaner
from daft_exprt.extract_features import extract_energy, extract_pitch, mel_spectrogram_HiFi, rescale_wav_to_float32
from daft_exprt.griffin_lim import griffin_lim_reconstruction_from_mel_spec
from daft_exprt.symbols import ascii, eos, punctuation, whitespace
from daft_exprt.utils import chunker, launch_multi_process, plot_2d_data


_logger = logging.getLogger(__name__)
FILE_ROOT = os.path.dirname(os.path.realpath(__file__))


def phonemize_sentence(sentence, hparams, log_queue):
    ''' Phonemize sentence using MFA
    '''
    # get MFA variables
    dictionary = hparams.mfa_dictionary
    g2p_model = hparams.mfa_g2p_model
    # load dictionary and extract word transcriptions
    word_trans = collections.defaultdict(list)
    with open(dictionary, 'r', encoding='utf-8') as f:
        lines = [line.strip().split() for line in f.readlines()] 
    for line in lines:
        word_trans[line[0].lower()].append(line[1:])
    # characters to consider in the sentence
    if hparams.language == 'english':
        all_chars = ascii + punctuation
    else:
        raise NotImplementedError()
    
    # clean sentence
    # "that's, an 'example! ' of a sentence. '"
    sentence = text_cleaner(sentence.strip(), hparams.language).lower().strip()
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
    sent_words.append(punctuation_end)
    
    # phonemize words and add word boundaries
    sentence_phonemized, unk_words = [], []
    while len(sent_words) != 0:
        word = sent_words.pop(0)
        if word in word_trans:
            phones = random.choice(word_trans[word])
            sentence_phonemized.append(phones)
        else:
            unk_words.append(word)
            sentence_phonemized.append('<unk>')
        # at this point we pass to the next word
        # we must add a word boundary between two consecutive words
        if len(sent_words) != 0:
            word_bound = sent_words.pop(0) if sent_words[0] in punctuation else whitespace
            sentence_phonemized.append(word_bound)
    # add EOS token
    sentence_phonemized.append(eos)
    
    # use MFA g2p model to phonemize unknown words
    if len(unk_words) != 0:
        rand_name = str(uuid.uuid4())
        oovs = os.path.join(FILE_ROOT, f'{rand_name}_oovs.txt')
        with open(oovs, 'w', encoding='utf-8') as f:
            for word in unk_words:
                f.write(f'{word}\n')
        # generate transcription for unknown words
        oovs_trans = os.path.join(FILE_ROOT, f'{rand_name}_oovs_trans.txt')
        tmp_dir = os.path.join(FILE_ROOT, f'{rand_name}')
        os.system(f'mfa g2p {g2p_model} {oovs} {oovs_trans} -t {tmp_dir}')
        # extract transcriptions
        with open(oovs_trans, 'r', encoding='utf-8') as f:
            lines = [line.strip().split() for line in f.readlines()]
        for line in lines:
            transcription = line[1:]
            unk_idx = sentence_phonemized.index('<unk>')
            sentence_phonemized[unk_idx] = transcription
        # remove files
        os.remove(oovs)
        os.remove(oovs_trans)
        rmtree(tmp_dir, ignore_errors=True)

    return sentence_phonemized


def save_mel_spec_plot_and_audio(item, output_dir, hparams, log_queue):
    ''' Save mel-outputs/alignment plots and generate an audio using Griffin-Lim algorithm

    :param item:                (n_mel_channels, T + 1) -- mel-spectrogram numpy array
    :param alignments:          (L, T + 1) -- alignment numpy array
    :param output_dir:          directory to save plots and audio
    :param file_name:           filename to save plots and audio
    :param hparams:             hyper-parameters used for pre-processing and training
    :param log_queue:           logging queue for multi-processing
    '''
    # create logger from logging queue
    qh = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    if not root.hasHandlers():
        root.setLevel(logging.INFO)
        root.addHandler(qh)
    logger = logging.getLogger(f"worker{str(uuid.uuid4())}")
    
    # extract items
    file_name, mel_spec, weight = item
    # create a figure from the output
    plot_2d_data(data=(mel_spec, weight),
                 x_labels=('Mel-Spec Prediction', 'Alignments'),
                 filename=os.path.join(output_dir, file_name + '.png'))
    # generate audio using Griffin-Lim
    waveform = griffin_lim_reconstruction_from_mel_spec(mel_spec, hparams, logger)
    if waveform != []:
        wavfile.write(os.path.join(output_dir, file_name + '.wav'), hparams.sampling_rate, waveform)


def collate_tensors(batch_sentences, batch_dur_factors, batch_energy_factors,
                    batch_pitch_factors, pitch_transform, batch_refs,
                    batch_speaker_ids, batch_file_names, hparams):
    ''' Extract PyTorch tensors for each sentence and collate them for batch generation
    '''
    # gather batch
    batch = []
    for sentence, dur_factors, energy_factors, pitch_factors, refs in \
        zip(batch_sentences, batch_dur_factors, batch_energy_factors, batch_pitch_factors, batch_refs):
            # encode input text as a sequence of int symbols
            symbols = []
            for item in sentence:
                if isinstance(item, list):  # correspond to phonemes of a word
                    symbols += [hparams.symbols.index(phone) for phone in item]
                else:  # correspond to word boundaries
                    symbols.append(hparams.symbols.index(item))
            symbols = torch.IntTensor(symbols)  # (L, )
            # extract duration factors
            if dur_factors is None:
                dur_factors = [1. for _ in range(len(symbols))]
            dur_factors = torch.FloatTensor(dur_factors)  # (L, )
            assert(len(dur_factors) == len(symbols)), \
                _logger.error(f'{len(dur_factors)} duration factors whereas there a {len(symbols)} symbols')
            # extract energy factors
            if energy_factors is None:
                energy_factors = [1. for _ in range(len(symbols))]
            energy_factors = torch.FloatTensor(energy_factors)  # (L, )
            assert(len(energy_factors) == len(symbols)), \
                _logger.error(f'{len(energy_factors)} energy factors whereas there a {len(symbols)} symbols')
            # extract pitch factors
            if pitch_factors is None:
                if pitch_transform == 'add':
                    pitch_factors = [0. for _ in range(len(symbols))]
                elif pitch_transform == 'multiply':
                    pitch_factors = [1. for _ in range(len(symbols))]
            pitch_factors = torch.FloatTensor(pitch_factors)  # (L, )
            assert(len(pitch_factors) == len(symbols)), \
                _logger.error(f'{len(pitch_factors)} pitch factors whereas there a {len(symbols)} symbols')
            # extract references
            refs = np.load(refs)
            energy_ref, pitch_ref, mel_spec_ref = refs['energy'], refs['pitch'], refs['mel_spec']
            energy_ref = torch.from_numpy(energy_ref).float()  # (T_ref, )
            pitch_ref = torch.from_numpy(pitch_ref).float()  # (T_ref, )
            mel_spec_ref = torch.from_numpy(mel_spec_ref).float()  # (n_mel_channels, T_ref)
            # gather data
            batch.append([symbols, dur_factors, energy_factors, pitch_factors, energy_ref, pitch_ref, mel_spec_ref])
    
    # find symbols sequence max length
    input_lengths, ids_sorted_decreasing = \
        torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
    max_input_len = input_lengths[0]
    # right pad sequences to max input length
    symbols = torch.LongTensor(len(batch), max_input_len).zero_()
    dur_factors = 1 + torch.FloatTensor(len(batch), max_input_len).zero_()
    energy_factors = 1 + torch.FloatTensor(len(batch), max_input_len).zero_()
    if pitch_transform == 'add':
        pitch_factors = torch.FloatTensor(len(batch), max_input_len).zero_()
    elif pitch_transform == 'multiply':
        pitch_factors = 1 + torch.FloatTensor(len(batch), max_input_len).zero_()
    # fill padded arrays
    for i in range(len(ids_sorted_decreasing)):
        # extract batch sequences
        symbols_seq = batch[ids_sorted_decreasing[i]][0]
        dur_factors_seq = batch[ids_sorted_decreasing[i]][1]
        energy_factors_seq = batch[ids_sorted_decreasing[i]][2]
        pitch_factors_seq = batch[ids_sorted_decreasing[i]][3]
        # add sequences to padded arrays
        symbols[i, :symbols_seq.size(0)] = symbols_seq
        dur_factors[i, :dur_factors_seq.size(0)] = dur_factors_seq
        energy_factors[i, :energy_factors_seq.size(0)] = energy_factors_seq
        pitch_factors[i, :pitch_factors_seq.size(0)] = pitch_factors_seq
    
    # find reference max length
    max_ref_len = max([x[6].size(1) for x in batch])
    # right zero-pad references to max output length
    energy_refs = torch.FloatTensor(len(batch), max_ref_len).zero_()
    pitch_refs = torch.FloatTensor(len(batch), max_ref_len).zero_()
    mel_spec_refs = torch.FloatTensor(len(batch), hparams.n_mel_channels, max_ref_len).zero_()
    ref_lengths = torch.LongTensor(len(batch))
    # fill padded arrays
    for i in range(len(ids_sorted_decreasing)):
        # extract batch sequences
        energy_ref_seq = batch[ids_sorted_decreasing[i]][4]
        pitch_ref_seq = batch[ids_sorted_decreasing[i]][5]
        mel_spec_ref_seq = batch[ids_sorted_decreasing[i]][6]
        # add sequences to padded arrays
        energy_refs[i, :energy_ref_seq.size(0)] = energy_ref_seq
        pitch_refs[i, :pitch_ref_seq.size(0)] = pitch_ref_seq
        mel_spec_refs[i, :, :mel_spec_ref_seq.size(1)] = mel_spec_ref_seq
        ref_lengths[i] = mel_spec_ref_seq.size(1)
    
    # reorganize speaker IDs and file names
    file_names = []
    speaker_ids = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
        file_names.append(batch_file_names[ids_sorted_decreasing[i]])
        speaker_ids[i] = batch_speaker_ids[ids_sorted_decreasing[i]]
    
    return symbols, dur_factors, energy_factors, pitch_factors, input_lengths, \
        energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids, file_names
    

def generate_batch_mel_specs(model, batch_sentences, batch_refs, batch_dur_factors,
                             batch_energy_factors, batch_pitch_factors, pitch_transform,
                             batch_speaker_ids, batch_file_names, output_dir, hparams,
                             n_jobs, use_griffin_lim=True):
    ''' Generate batch mel-specs using Daft-Exprt
    '''
    # add speaker info to file name
    for idx, file_name in enumerate(batch_file_names):
        file_name += f'_spk_{batch_speaker_ids[idx]}'
        file_name += f'_ref_{os.path.basename(batch_refs[idx]).replace(".npz", "")}'
        batch_file_names[idx] = file_name
        _logger.info(f'Generating "{batch_sentences[idx]}" as "{file_name}"')
    # collate batch tensors
    symbols, dur_factors, energy_factors, pitch_factors, input_lengths, \
        energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids, file_names = \
            collate_tensors(batch_sentences, batch_dur_factors, batch_energy_factors,
                            batch_pitch_factors, pitch_transform, batch_refs,
                            batch_speaker_ids, batch_file_names, hparams)
    # put tensors on GPU
    gpu = next(model.parameters()).device
    symbols = symbols.cuda(gpu, non_blocking=True).long()  # (B, L_max)
    dur_factors = dur_factors.cuda(gpu, non_blocking=True).float()  # (B, L_max)
    energy_factors = energy_factors.cuda(gpu, non_blocking=True).float()  # (B, L_max)
    pitch_factors = pitch_factors.cuda(gpu, non_blocking=True).float()  # (B, L_max)
    input_lengths = input_lengths.cuda(gpu, non_blocking=True).long()  # (B, )
    energy_refs = energy_refs.cuda(gpu, non_blocking=True).float()  # (B, T_max)
    pitch_refs = pitch_refs.cuda(gpu, non_blocking=True).float()  # (B, T_max)
    mel_spec_refs = mel_spec_refs.cuda(gpu, non_blocking=True).float()  # (B, n_mel_channels, T_max)
    ref_lengths = ref_lengths.cuda(gpu, non_blocking=True).long()  # (B, )
    speaker_ids = speaker_ids.cuda(gpu, non_blocking=True).long()  # (B, )
    # perform inference
    inputs = (symbols, dur_factors, energy_factors, pitch_factors, input_lengths,
              energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids)
    try:
        encoder_preds, decoder_preds, alignments = model.inference(inputs, pitch_transform, hparams)
    except:
        encoder_preds, decoder_preds, alignments = model.module.inference(inputs, pitch_transform, hparams)
    # parse outputs
    duration_preds, durations_int, energy_preds, pitch_preds, input_lengths = encoder_preds
    mel_spec_preds, output_lengths = decoder_preds
    weights = alignments
    # transfer data to cpu and convert to numpy array
    duration_preds = duration_preds.detach().cpu().numpy()  # (B, L_max)
    durations_int = durations_int.detach().cpu().numpy()  # (B, L_max)
    energy_preds = energy_preds.detach().cpu().numpy()  # (B, L_max)
    pitch_preds = pitch_preds.detach().cpu().numpy()  # (B, L_max)
    input_lengths = input_lengths.detach().cpu().numpy()  # (B, )
    mel_spec_preds = mel_spec_preds.detach().cpu().numpy()  # (B, n_mel_channels, T_max)
    output_lengths = output_lengths.detach().cpu().numpy()  # (B)
    weights = weights.detach().cpu().numpy()  # (B, L_max, T_max)
    
    # save preds for each element in the batch
    predictions = {}
    for line_idx in range(mel_spec_preds.shape[0]):
        # crop prosody preds to the correct length
        duration_pred = duration_preds[line_idx, :input_lengths[line_idx]]  # (L, )
        duration_int = durations_int[line_idx, :input_lengths[line_idx]]  # (L, )
        energy_pred = energy_preds[line_idx, :input_lengths[line_idx]]  # (L, )
        pitch_pred = pitch_preds[line_idx, :input_lengths[line_idx]]  # (L, )
        # crop mel-spec to the correct length
        mel_spec_pred = mel_spec_preds[line_idx, :, :output_lengths[line_idx]]  # (n_mel_channels, T)
        # crop weights to the correct length
        weight = weights[line_idx, :input_lengths[line_idx], :output_lengths[line_idx]]
        # save generated spectrogram
        file_name = file_names[line_idx]
        np.savez(os.path.join(output_dir, f'{file_name}.npz'), mel_spec=mel_spec_pred)
        # store predictions 
        predictions[f'{file_name}'] = [duration_pred, duration_int, energy_pred, pitch_pred, mel_spec_pred, weight]
    
    # save plots and generate audio using Griffin-Lim
    if use_griffin_lim:
        items = [[file_name, mel_spec, weight] for file_name, (_, _, _, _, mel_spec, weight) in predictions.items()]
        launch_multi_process(iterable=items, func=save_mel_spec_plot_and_audio, n_jobs=n_jobs,
                             timer_verbose=False, output_dir=output_dir, hparams=hparams)
    
    return predictions


def generate_mel_specs(model, sentences, file_names, speaker_ids, refs, output_dir, hparams,
                       dur_factors=None, energy_factors=None, pitch_factors=None, batch_size=1,
                       n_jobs=1, use_griffin_lim=False, get_time_perf=False):
    ''' Generate mel-specs using Daft-Exprt

        sentences = [
            sentence_1,
            ...
            sentence_N
        ]
        each sentence is a list of symbols:
            sentence = [
                [symbols_word_1],
                word_boundary_symbol,
                [symbols_word_2],
                word_boundary_symbol,
                ...
            ]
        for example, here is a sentence of 5 words, 6 word boundaries and a total of 17 symbols:
            sentence = [['IH0', 'Z'], ' ', ['IH0', 'T'], ',', ['AH0'], ' ', ['G', 'UH1', 'D'], ' ', ['CH', 'OY1', 'S'], '?', '~']
        
        file_names = [
            file_name_1,
            ...
            file_name_N
        ]
        
        speaker_ids = [
            speaker_id_1,
            ...
            speaker_id_N
        ]
        
        refs = [
            path_to_ref_1.npz,
            ...
            path_to_ref_N.npz
        ]
        
        dur_factors = [
            [factor_sentence_1_symbol_1, factor_sentence_1_symbol_2, ...],
            ...
            [factor_sentence_N_symbol_1, factor_sentence_N_symbol_2, ...]
        ]
        if None, duration predictions are not modified
        
        energy_factors = [
            [factor_sentence_1_symbol_1, factor_sentence_1_symbol_2, ...],
            ...
            [factor_sentence_N_symbol_1, factor_sentence_N_symbol_2, ...]
        ]
        if None, energy predictions are not modified
        
        pitch_factors = [
            "transform",
            [
                [factor_sentence_1_symbol_1, factor_sentence_1_symbol_2, ...],
                ...
                [factor_sentence_N_symbol_1, factor_sentence_N_symbol_2, ...]
            ] 
        ]
        There are 2 types of transforms:
            - pitch shift: "add"
            - pitch multiply: "multiply"
        if None, pitch predictions are not modified
    '''
    # set default values if prosody factors are None
    dur_factors = [None for _ in range(len(sentences))] if dur_factors is None else dur_factors
    energy_factors = [None for _ in range(len(sentences))] if energy_factors is None else energy_factors
    pitch_factors = ['add', [None for _ in range(len(sentences))]] if pitch_factors is None else pitch_factors
    # get pitch transform
    pitch_transform = pitch_factors[0].lower()
    pitch_factors = pitch_factors[1]
    assert(pitch_transform in ['add', 'multiply']), _logger.error(f'Pitch transform "{pitch_transform}" is not currently supported')
    # check lists have the same size
    assert (len(file_names) == len(sentences)), _logger.error(f'{len(file_names)} filenames but there are {len(sentences)} sentences to generate')
    assert (len(speaker_ids) == len(sentences)), _logger.error(f'{len(speaker_ids)} speaker IDs but there are {len(sentences)} sentences to generate')
    assert (len(refs) == len(sentences)), _logger.error(f'{len(refs)} references but there are {len(sentences)} sentences to generate')
    assert (len(dur_factors) == len(sentences)), _logger.error(f'{len(dur_factors)} duration factors but there are {len(sentences)} sentences to generate')
    assert (len(energy_factors) == len(sentences)), _logger.error(f'{len(energy_factors)} energy factors but there are {len(sentences)} sentences to generate')
    assert (len(pitch_factors) == len(sentences)), _logger.error(f'{len(pitch_factors)} pitch factors but there are {len(sentences)} sentences to generate')
      
    # we don't need computational graph for inference
    model.eval()  # set eval mode
    os.makedirs(output_dir, exist_ok=True)
    predictions, time_per_batch = {}, []
    with torch.no_grad():
        for batch_sentences, batch_refs, batch_dur_factors, batch_energy_factors, \
            batch_pitch_factors, batch_speaker_ids, batch_file_names in \
                zip(chunker(sentences, batch_size), chunker(refs, batch_size), 
                    chunker(dur_factors, batch_size), chunker(energy_factors, batch_size),
                    chunker(pitch_factors, batch_size), chunker(speaker_ids, batch_size),
                    chunker(file_names, batch_size)):
                sentence_begin = time.time() if get_time_perf else None
                batch_predictions =  generate_batch_mel_specs(model, batch_sentences, batch_refs, batch_dur_factors,
                                                              batch_energy_factors, batch_pitch_factors, pitch_transform,
                                                              batch_speaker_ids, batch_file_names, output_dir, hparams,
                                                              n_jobs, use_griffin_lim)
                predictions.update(batch_predictions)
                time_per_batch += [time.time() - sentence_begin] if get_time_perf else []
    
    # display overall time performance
    if get_time_perf:
        # get duration of each sentence
        durations = []
        for prediction in predictions.values():
            _, _, _, _, mel_spec, _ = prediction
            nb_frames = mel_spec.shape[1]
            nb_wav_samples = (nb_frames - 1) * hparams.hop_length + hparams.filter_length
            if hparams.centered:
                nb_wav_samples -= 2 * int(hparams.filter_length / 2)
            duration = nb_wav_samples / hparams.sampling_rate
            durations.append(duration)
        _logger.info(f'')
        _logger.info(f'{len(predictions)} sentences ({sum(durations):.2f}s) generated in {sum(time_per_batch):.2f}s')
        _logger.info(f'DaftExprt RTF: {sum(durations)/sum(time_per_batch):.2f}')
    
    return predictions


def extract_reference_parameters(audio_ref, output_dir, hparams):
    ''' Extract energy, pitch and mel-spectrogram parameters from audio
        Save numpy arrays to .npz file
    '''
    # check if file name already exists
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(audio_ref).replace('.wav', '')
    ref_file = os.path.join(output_dir, f'{file_name}.npz')
    if not os.path.isfile(ref_file):
        # read wav file to range [-1, 1] in np.float32
        wav, fs = librosa.load(audio_ref, sr=hparams.sampling_rate)
        wav = rescale_wav_to_float32(wav)
        # get log pitch
        pitch = extract_pitch(wav, fs, hparams)
        # extract mel-spectrogram
        mel_spec = mel_spectrogram_HiFi(wav, hparams)
        # get energy
        energy = extract_energy(np.exp(mel_spec))
        # check sizes are correct
        assert(len(pitch) == mel_spec.shape[1]), f'{len(pitch)} -- {mel_spec.shape[1]}'
        assert(len(energy) == mel_spec.shape[1]), f'{len(energy)} -- {mel_spec.shape[1]}'
        # save references to .npz file
        np.savez(ref_file, energy=energy, pitch=pitch, mel_spec=mel_spec)


def prepare_sentences_for_inference(text_file, output_dir, hparams, n_jobs):
    ''' Phonemize and format sentences to synthesize
    '''
    # create output directory or delete everything if it already exists
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    
    # extract sentences to synthesize
    assert(os.path.isfile(text_file)), _logger.error(f'There is no such file {text_file}')
    with open(os.path.join(text_file), 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    file_names = [f'{os.path.basename(text_file)}_line{idx}' for idx in range(len(sentences))]
    # phonemize
    hparams.update_mfa_paths()
    sentences = launch_multi_process(iterable=sentences, func=phonemize_sentence,
                                     n_jobs=n_jobs, timer_verbose=False, hparams=hparams)

    # save the sentences in a file
    with open(os.path.join(output_dir, 'sentences_to_generate.txt'), 'w', encoding='utf-8') as f:
        for sentence, file_name in zip(sentences, file_names):
            text = ''
            for item in sentence:
                if isinstance(item, list):  # corresponds to phonemes of a word
                    item = '{' + ' '.join(item) + '}'
                text = f'{text} {item} '
            text = collapse_whitespace(text).strip()
            f.write(f'{file_name}|{text}\n')    

    return sentences, file_names
