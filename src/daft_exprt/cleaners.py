import re

from unidecode import unidecode

from daft_exprt.normalize_numbers import normalize_numbers


'''
Cleaners are transformations that need to be applied to in-the-wild text before it is sent to the acoustic model

greatly inspired from https://github.com/keithito/tacotron
'''

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def hyphen_remover(text):
    text = re.sub('–', ', ', text)
    text = re.sub(' -- ', ', ', text)
    return re.sub('-', ' ', text)


def quote_remover(text):
    return re.sub('"', '', text)


def parenthesis_remover(text):
    return re.sub('\(|\)', '', text)


def space_coma_replacer(text):
    return re.sub('[\s,]*,+[\s,]*', ', ', text)


def incorrect_starting_character_remover(text):
    while text.startswith((',', ' ', '.', '!', '?', '-')):
        text = text[1:]
    return text


def apostrophee_formater(text):
    return re.sub('’', '\'', text)


def dot_coma_replacer(text):
    return re.sub(';', ',', text)


def double_dot_replacer(text):
    return re.sub(':', ',', text)


def underscore_replacer(text):
    return re.sub('_', ' ', text)


def triple_dot_replacer(text):
    text = re.sub('…', '.', text)
    return re.sub('[\s\.]*\.+[\s\.]*', '. ', text)


def multiple_punctuation_fixer(text):
    text = re.sub('[\s\.,?!]*\?+[\s\.,?!]*', '? ', text)
    text = re.sub('[\s\.,!]*\!+[\s\.,!]*', '! ', text)
    return re.sub('[\s\.,]*\.+[\s\.,]*', '. ', text)


def english_cleaners(text):
    ''' pipeline for English text, including number and abbreviation expansion

    :param text: sentence to process
    '''
    # convert to regular english letters in lowercase.
    text = convert_to_ascii(text)
    text = lowercase(text)

    # replace all abbreviations and numbers with text
    text = expand_numbers(text)
    text = expand_abbreviations(text)

    # deal with punctuation
    text = hyphen_remover(text)
    text = quote_remover(text)
    text = dot_coma_replacer(text)  # replace by a coma
    text = double_dot_replacer(text)  # replace by a coma
    text = triple_dot_replacer(text)  # replace by a coma
    text = apostrophee_formater(text)
    text = parenthesis_remover(text)
    text = space_coma_replacer(text)
    text = underscore_replacer(text)
    text = collapse_whitespace(text)
    text = incorrect_starting_character_remover(text)
    text = multiple_punctuation_fixer(text)
    text = text.strip()

    return text


def text_cleaner(text, lang='english'):
    if lang.lower() == 'english':
        text = english_cleaners(text)

    return text
