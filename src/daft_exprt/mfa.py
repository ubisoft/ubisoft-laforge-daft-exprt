import logging
import logging.handlers
import os
import uuid

import tgt

from shutil import move, rmtree

from daft_exprt.cleaners import text_cleaner
from daft_exprt.symbols import MFA_SIL_WORD_SYMBOL, MFA_SIL_PHONE_SYMBOLS, MFA_UNK_WORD_SYMBOL, \
    MFA_UNK_PHONE_SYMBOL, SIL_WORD_SYMBOL, SIL_PHONE_SYMBOL
from daft_exprt.utils import launch_multi_process


_logger = logging.getLogger(__name__)


''' 
    Align speaker corpuses using MFA
    https://montreal-forced-aligner.readthedocs.io/en/latest/
'''


def move_file(file, src_dir, dst_dir, log_queue):
    ''' Dummy function to move a file in multi-processing mode
    '''
    move(os.path.join(src_dir, file), os.path.join(dst_dir, file))


def prepare_corpus(corpus_dir, language):
    ''' Prepare corpus for MFA
        Create .lab files for each audio file
    '''
    # check wavs directory and speaker metadata file exist
    wavs_dir = os.path.join(corpus_dir, 'wavs')
    metadata = os.path.join(corpus_dir, 'metadata.csv')
    assert(os.path.isdir(wavs_dir)), _logger.error(f'There is no such directory: {wavs_dir}')
    assert(os.path.isfile(metadata)), _logger.error(f'There is no such file: {metadata}')
    
    # extract lines from metadata.csv
    with open(metadata, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [x.strip().split(sep='|') for x in lines]  # [[file_name, text], ...]
    # check there is only 1 pipe "|" separator
    for line in lines:
        assert(len(line) == 2), _logger.error(f'Problem in metadata file: {corpus_dir} -- Line: {line}')
    # extract file names and corresponding text
    file_names = [line[0].strip() for line in lines]
    texts = [line[1].strip() for line in lines]
    
    # create .lab file for each audio file
    wavs = [os.path.join(wavs_dir, x) for x in os.listdir(wavs_dir) if x.endswith('.wav')]
    for wav in wavs:
        # search metadata lines associated to wav file
        wav_name = os.path.basename(wav).replace('.wav', '').strip()
        lines_idx = [idx for idx, file_name in enumerate(file_names) if wav_name == file_name]
        # only create .lab if ONE line is associated to wav file
        if len(lines_idx) == 1:
            # get corresponding text and clean it
            text = texts[lines_idx[0]]
            text = text_cleaner(text, language).strip()
            # save it to .lab file
            with open(os.path.join(wavs_dir, f'{wav_name}.lab'), 'w', encoding='utf-8') as f:
                f.write(text)
        # remove lines for computational efficiency
        for i, idx in enumerate(lines_idx):
            del file_names[idx - i]
            del texts[idx - i]


def _extract_markers(text_grid_file, log_queue):
    ''' Extract word/phone alignment markers from .TextGrid file
    '''
    # create logger from logging queue
    qh = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    if not root.hasHandlers():
        root.setLevel(logging.INFO)
        root.addHandler(qh)
    logger = logging.getLogger(f"worker{str(uuid.uuid4())}")
    
    # load text grid
    text_grid = tgt.io.read_textgrid(text_grid_file, include_empty_intervals=True)
    # extract word and phone tiers
    words_tier = text_grid.get_tier_by_name("words")
    words = [[word.start_time, word.end_time, word.text] for word in words_tier._objects]
    phones_tier = text_grid.get_tier_by_name("phones")
    phones = [[phone.start_time, phone.end_time, phone.text] for phone in phones_tier._objects]
    # set silence symbol according to chosen nomenclature
    for marker in words:
        _, _, word = marker
        if word == MFA_SIL_WORD_SYMBOL:
            marker[-1] = SIL_WORD_SYMBOL
    for marker in phones:
        _, _, phone = marker
        if phone in MFA_SIL_PHONE_SYMBOLS:
            marker[-1] = SIL_PHONE_SYMBOL
    # merge subsequent silences on phoneme level
    # for example, it is possible to have: AH0 - SIL - SIL - OW0
    # this should be merged as follows: AH0 - SIL - OW0
    phones_old = phones.copy()
    phones = [phones_old[0]]
    for marker in phones_old[1:]:
        _, end, phone = marker
        prev_phone = phones[-1][2]
        if prev_phone == phone == SIL_PHONE_SYMBOL:
            phones[-1][1] = end
        else:
            phones.append(marker)
    
    # gather words and phones markers together
    # ignore if an unknown word/phone is detected
    # ignore if a silence is detected withing the word
    silence_error = False
    all_words = [word for _, _, word in words]
    all_phones = [phone for _, _, phone in phones]
    if MFA_UNK_WORD_SYMBOL not in all_words and MFA_UNK_PHONE_SYMBOL not in all_phones:
        markers = []
        for word_idx, word_marker in enumerate(words):
            begin_word, end_word, word = word_marker
            for phone_marker in phones:
                begin_phone, end_phone, phone = phone_marker
                if begin_word <= begin_phone and end_phone <= end_word:
                    # check silent word and phoneme have a one to one correspondance
                    if word == SIL_WORD_SYMBOL:
                        assert(phone == SIL_PHONE_SYMBOL and begin_word == begin_phone and end_word == end_phone), \
                            logger.error(f'{text_grid_file} -- error with silence -- word number {word_idx}')
                    else:  # check there are no silence errors
                        if phone == SIL_PHONE_SYMBOL:
                            logger.warning(f'{text_grid_file} -- silence within word -- word number {word_idx} -- Ignoring file')
                            silence_error = True
                    # add to list
                    markers.append([f'{begin_phone:.3f}', f'{end_phone:.3f}', phone, word, str(word_idx)])
                else:
                    # check phone does not overlap with word
                    assert(end_phone <= begin_word or end_word <= begin_phone), \
                        logger.error(f'{text_grid_file} -- word and phoneme overlap -- word number {word_idx}')
        
        if not silence_error:
            # trim leading and tailing silences
            phone_lead, phone_tail = markers[0][2], markers[-1][2]
            if phone_lead == SIL_PHONE_SYMBOL:
                markers.pop(0)
            if phone_tail == SIL_PHONE_SYMBOL:
                markers.pop(-1)
            # check everything is correct with trimming
            phone_lead, phone_tail = markers[0][2], markers[-1][2]
            assert(phone_lead != SIL_PHONE_SYMBOL and phone_tail != SIL_PHONE_SYMBOL), \
                logger.error(f'{text_grid_file} -- problem with sentence triming')
            # check timings are correct
            for marker_curr, marker_next in zip(markers[:-1], markers[1:]):
                begin_curr, end_curr =  marker_curr[0], marker_curr[1]
                begin_next, end_next =  marker_next[0], marker_next[1]
                assert(float(end_curr) == float(begin_next)), logger.error(f'{text_grid_file} -- problem with sentence timing')
                assert(float(begin_curr) < float(end_curr)), logger.error(f'{text_grid_file} -- problem with sentence timing')
                assert(float(begin_next) < float(end_next)), logger.error(f'{text_grid_file} -- problem with sentence timing')
            
            # save file in .markers format
            text_grid_dir = os.path.dirname(text_grid_file)
            file_name = os.path.basename(text_grid_file).replace('.TextGrid', '')
            with open(os.path.join(text_grid_dir, f'{file_name}.markers'), 'w', encoding='utf-8') as f:
                f.writelines(['\t'.join(x) + '\n' for x in markers])


def extract_markers(text_grid_dir, n_jobs):
    ''' Extract word/phone alignment markers from .TextGrid files contained in TextGrid directory
    '''
    # get all .TextGrid files contained in the directory that do not have .markers files
    all_grid_files = [os.path.join(text_grid_dir, x) for x in os.listdir(text_grid_dir) if x.endswith('.TextGrid')]
    grid_files_to_process = [x for x in all_grid_files if not os.path.isfile(x.replace('.TextGrid', '.markers'))]
    _logger.info(f'Folder: {text_grid_dir} -- {len(all_grid_files) - len(grid_files_to_process)} TextGrid files already processed -- '
                 f'{len(grid_files_to_process)} TextGrid files need to be processed')
    
    # extract markers for words and phones
    launch_multi_process(iterable=grid_files_to_process, func=_extract_markers, n_jobs=n_jobs, timer_verbose=False)


def mfa(dataset_dir, hparams, n_jobs):
    ''' Run MFA on every speaker data set and extract timing markers for words and phones
    '''
    _logger.info('--' * 30)
    _logger.info('Running MFA for each speaker data set'.upper())
    _logger.info('--' * 30)
    
    # perform alignment for each speaker
    for speaker in hparams.speakers:
        _logger.info(f'Speaker: "{speaker}"')
        # check if alignment has already been performed
        corpus_dir = os.path.join(dataset_dir, speaker)
        align_out_dir = os.path.join(corpus_dir, 'align')
        if not os.path.isdir(align_out_dir):
            # initialize variables
            language = hparams.language
            dictionary = hparams.mfa_dictionary
            g2p_model = hparams.mfa_g2p_model
            acoustic_model = hparams.mfa_acoustic_model
            temp_dir = os.path.join(corpus_dir, 'tmp')
            
            # create .lab files for each audio file
            _logger.info('Preparing MFA corpus')
            prepare_corpus(corpus_dir, language)
            
            # # uncomment if you need to validate your corpus before MFA alignment
            # # validate corpuses to ensure there are no issues with the data format
            # _logger.info('Validating corpus')
            # tmp_dir = os.path.join(temp_dir, 'validate')
            # os.system(f'mfa validate {corpus_dir} {dictionary} '
            #           f'{acoustic_model} -t {tmp_dir} -j {n_jobs}')
            # # use a G2P model to generate a pronunciation dictionary for unknown words
            # # this can later be added manually to the dictionary
            # oovs = os.path.join(tmp_dir, os.path.basename(speaker), 'corpus_data', 'oovs_found.txt')
            # if os.path.isfile(oovs):
            #     _logger.info('Generating transcriptions for unknown words')
            #     oovs_trans = os.path.join(corpus_dir, 'oovs_transcriptions.txt')
            #     os.system(f'mfa g2p {g2p_model} {oovs} {oovs_trans}')
            
            # perform forced alignment with a pretrained acoustic model
            _logger.info('Performing forced alignment using a pretrained model')
            tmp_dir = os.path.join(temp_dir, 'align')
            os.system(f'mfa align {corpus_dir} {dictionary} {acoustic_model} '
                        f'{align_out_dir} -t {tmp_dir} -j {n_jobs} -v -c')
            
            # extract word/phone alignment markers from .TextGrid files
            _logger.info('Extracting markers')
            text_grid_dir = os.path.join(align_out_dir, 'wavs')
            assert(os.path.isdir(text_grid_dir)), _logger.error(f'There is no such dir {text_grid_dir}')
            all_files = [x for x in os.listdir(text_grid_dir)]
            launch_multi_process(iterable=all_files, func=move_file, n_jobs=n_jobs,
                                 src_dir=text_grid_dir, dst_dir=align_out_dir, timer_verbose=False)
            rmtree(text_grid_dir, ignore_errors=True)
            extract_markers(align_out_dir, n_jobs)
            # move .lab files to markers dir
            _logger.info('Moving .lab files to markers directory')
            wavs_dir = os.path.join(corpus_dir, 'wavs')
            lab_files = [x for x in os.listdir(wavs_dir) if x.endswith('.lab')]
            launch_multi_process(iterable=lab_files, func=move_file, n_jobs=n_jobs,
                                 src_dir=wavs_dir, dst_dir=align_out_dir, timer_verbose=False)
            # remove temp dir
            rmtree(temp_dir, ignore_errors=True)
            # display stats
            wavs = [x for x in os.listdir(wavs_dir) if x.endswith('.wav')]
            markers = [x for x in os.listdir(align_out_dir) if x.endswith('.markers')]
            _logger.info(f'{len(markers) / len(wavs) * 100:.2f}% of the data set aligned')
        else:
            # extract word/phone alignment markers from .TextGrid files
            _logger.info('MFA alignment already performed')
            _logger.info('Extracting markers')
            extract_markers(align_out_dir, n_jobs)
            # display stats
            wavs_dir = os.path.join(corpus_dir, 'wavs')
            wavs = [x for x in os.listdir(wavs_dir) if x.endswith('.wav')]
            markers = [x for x in os.listdir(align_out_dir) if x.endswith('.markers')]
            _logger.info(f'{len(markers) / len(wavs) * 100:.2f}% of the data set aligned')
        _logger.info('')
