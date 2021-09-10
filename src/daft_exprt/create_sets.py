import logging
import os


_logger = logging.getLogger(__name__)


def create_sets(features_dir, hparams, proportion_validation=0.1):
    ''' Create train and validation sets, for all specified speakers

    :param features_dir:                directory containing all the speakers features files
    :param hparams:                     hyper-parameters used for pre-processing
    :param proportion_validation:       for each speaker, proportion of examples (%) that will be in the validation set
    '''
    # create directory where extracted train/validation sets will be saved
    os.makedirs(os.path.dirname(hparams.training_files), exist_ok=True)
    os.makedirs(os.path.dirname(hparams.validation_files), exist_ok=True)
    # create train/validation text files
    file_training = open(hparams.training_files, 'w', encoding='utf-8')
    file_validation = open(hparams.validation_files, 'w', encoding='utf-8')

    # iterate over speakers
    _logger.info('--' * 30)
    _logger.info('Creating training and validation sets'.upper())
    _logger.info('--' * 30)
    for speaker, speaker_id in zip(hparams.speakers, hparams.speakers_id):
        _logger.info(f'Speaker: "{speaker}" -- ID: {speaker_id} -- Validation files: {proportion_validation}%')
        # check metadata file exists
        spk_features_dir = os.path.join(features_dir, speaker)
        metadata = os.path.join(spk_features_dir, 'metadata.csv')
        # read metadata lines
        with open(metadata, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [x.strip().split(sep='|') for x in lines]  # [[file_name, text], ...]
        # get available features files for training
        # some metadata files might miss because there was no .markers associated to the file
        file_names = [line[0].strip() for line in lines]
        features_files = [x for x in file_names if os.path.isfile(os.path.join(spk_features_dir, f'{x}.npy'))]
        nb_feats_files = len(features_files)
        
        ctr = 0
        validation_ctr = 0
        for feature_file in features_files:
            # store the line
            ctr += 1
            new_line = f'{spk_features_dir}|{feature_file}|{speaker_id}\n'
            if ctr % int(100 / proportion_validation) == 0 or (ctr == nb_feats_files and validation_ctr == 0):
                file_validation.write(new_line)
                validation_ctr += 1
            else:
                file_training.write(new_line)   
        _logger.info('')

    file_training.close()
    file_validation.close()
