import string


# silence symbols and unknown word symbols used by MFA in ".TextGrid" files
MFA_SIL_WORD_SYMBOL = ''
MFA_SIL_PHONE_SYMBOLS = ['', 'sp', 'sil']
MFA_UNK_WORD_SYMBOL = '<unk>'
MFA_UNK_PHONE_SYMBOL = 'spn'

# silence symbols used in ".markers" files
# allows to only have 1 silence symbol instead of 3
SIL_WORD_SYMBOL = '<sil>'
SIL_PHONE_SYMBOL = 'SIL'

# PAD and EOS token
pad = '_'
eos = '~'

# whitespace character
whitespace = ' '

# punctuation to consider in input sentence
punctuation = ',.!?'

# Arpabet stressed phonetic set
arpabet_stressed = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0',
                    'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1',
                    'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH',
                    'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH',
                    'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

# ascii letters
ascii = string.ascii_lowercase.upper() + string.ascii_lowercase

# symbols used by Daft-Exprt in english language
symbols_english = list(pad + eos + whitespace + punctuation) + arpabet_stressed
