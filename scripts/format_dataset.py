import argparse
import logging
import os

from shutil import copyfile


_logger = logging.getLogger(__name__)


'''
    This script modifies speakers data sets to match the required format
    Each speaker data set must be of the following format:
    
    /speaker_name
        metadata.csv
        /wavs
            wav_file_name_1.wav
            wav_file_name_2.wav
            ...
    
    metadata.csv must be formatted as follows (pipe "|" separator):
        wav_file_name_1|text_1
        wav_file_name_2|text_2
        ...
'''


def format_LJ_speech(lj_args):
    ''' Format LJ data set
        Only metadata.csv needs to be modified
    '''
    # read metadata lines
    _logger.info('Formatting LJ Speech')
    metadata = os.path.join(lj_args.data_set_dir, 'metadata.csv')
    assert(os.path.isfile(metadata)), _logger.error(f'There is no such file {metadata}')
    with open(metadata, 'r', encoding='utf-8') as f:
        metadata_lines = f.readlines()
    # create new metadata.csv
    metadata_lines = [line.strip().split(sep='|') for line in metadata_lines]
    metadata_lines = [f'{line[0]}|{line[2]}\n' for line in metadata_lines]
    with open(metadata, 'w', encoding='utf-8') as f:
        f.writelines(metadata_lines)
    _logger.info('Done!')


def format_ESD(esd_args):
    ''' Format ESD data set
    '''
    # extract speaker dirs depending on the language
    _logger.info(f'Formatting ESD -- Language = {esd_args.language}')
    speakers = [x for x in os.listdir(esd_args.data_set_dir) if 
                os.path.isdir(os.path.join(esd_args.data_set_dir, x))]
    speakers.sort()
    if esd_args.language == 'english':
        for speaker in speakers[10:]:
            _logger.info(f'Speaker -- {speaker}')
            speaker_dir = os.path.join(esd_args.data_set_dir, speaker)
            spk_out_dir = os.path.join(esd_args.data_set_dir, esd_args.language, speaker)
            os.makedirs(spk_out_dir, exist_ok=True)
            # read metadata lines
            if speaker == speakers[10]:
                metadata = os.path.join(speaker_dir,f'{speaker}.txt')
                assert(os.path.isfile(metadata)), _logger.error(f'There is no such file {metadata}')
                with open(metadata, 'r', encoding='utf-8') as f:
                    metadata_lines = f.readlines()
                metadata_lines = [line.strip().split(sep='\t') for line in metadata_lines]
            # create new metadata.csv
            spk_metadata_lines = [f'{speaker}_{line[0].strip().split(sep="_")[1]}|{line[1]}\n'
                                  for line in metadata_lines]
            with open(os.path.join(spk_out_dir, 'metadata.csv'), 'w', encoding='utf-8') as f:
                f.writelines(spk_metadata_lines)
            # copy all audio files to /wavs directory
            wavs_dir = os.path.join(spk_out_dir, 'wavs')
            os.makedirs(wavs_dir, exist_ok=True)
            for root, _, files in os.walk(speaker_dir):
                wav_files = [x for x in files if x.endswith('.wav')]
                for wav_file in wav_files:
                    src = os.path.join(root, wav_file)
                    dst = os.path.join(wavs_dir, wav_file)
                    copyfile(src, dst)
    elif esd_args.language == 'mandarin':
        _logger.error(f'"mandarin" not implemented')
    else:
        _logger.error(f'"language" must be either "english" or "mandarin", not "{esd_args.language}"')
    _logger.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to format speakers data sets')
    subparsers = parser.add_subparsers(help='commands for targeting a specific data set')

    parser.add_argument('-dd', '--data_set_dir', type=str,
                        help='path to the directory containing speakers data sets to format')
    
    parser_LJ = subparsers.add_parser('LJ', help='format LJ data set')
    parser_LJ.set_defaults(func=format_LJ_speech)
    
    parser_ESD = subparsers.add_parser('ESD', help='format emotional speech dataset from Zhou et al.')
    parser_ESD.set_defaults(func=format_ESD)
    parser_ESD.add_argument('-lg', '--language', type=str,
                            help='either english or mandarin')
    
    args = parser.parse_args()
    
    # set logger config
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
        ],
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )

    # run args
    args.func(args)
