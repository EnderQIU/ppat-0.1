import os
import string

# All constants and Configurations should be listed in this file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CMU Origin Dictionary Path
CMU_DICT_PATH = os.path.join(BASE_DIR, 'dataset', 'cmudict-0.7b')
CMU_SYMBOLS_PATH = os.path.join(BASE_DIR, 'dataset', 'cmudict-0.7b.symbols')
ABBREV_CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'abbreviations.csv')
QTY_ABBREV_CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'quantity_abbreviations.csv')
# Output from pickle
ABBREV_DICT_PATH = os.path.join(BASE_DIR, 'output', 'bp-abbrev-dict.pkl')
QTY_ABBREV_DICT_PATH = os.path.join(BASE_DIR, 'output', 'bp-qty-abbrev-dict.pkl')
PHONETIC_DICT_PATH = os.path.join(BASE_DIR, 'output', 'bp-phonetic-dict.pkl')
PHONETIC_SYMBOLS_PATH = os.path.join(BASE_DIR, 'output', 'bp-phonetic-symbols.pkl')
# Model Savings
PREDICTION_MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'output', 'prediction_model_weights.hdf5')
# Utils
ALLOWED_SYMBOLS = [".", "-", "'"]
ALLOWED_CHARS = ALLOWED_SYMBOLS + list(string.ascii_uppercase)
START_PHONE_SYM = '\t'
END_PHONE_SYM = '\n'
MIN_WORD_LEN = 2
MAX_WORD_LEN = 20
MIN_PRONUNCIATION_LEN = 2
MAX_PRONUNCIATION_LEN = 20
MAX_PHONE_SEQ_LEN = 19
# Phone Sequences are padded with Start/End tokens
MAX_PADDED_PHONE_SEQ_LEN = MAX_PHONE_SEQ_LEN + 2

# Model Config
# Random (100*TEST_SCALE)% of all data will be used
DATA_LIMIT_SCALE = 0.8
# Ahead (100*TEST_SCALE)% of data (already suffled) will be used for training, and the rest for evaluating
TRAIN_SCALE = 0.5
# Random (100*TEST_SCALE)% of train data will be used for testing
TEST_SCALE = 0.2

# Variable constants
remove_digits = str.maketrans('', '', string.digits)
# A-Z
letters = list(string.ascii_uppercase)
# All phonetics
stressable_phonetics = [
    "AA", "AA0", "AA1", "AA2",  # 0 1 2 3
    "AE", "AE0", "AE1", "AE2",  # index // 4 = index of non-stressed-phonetic text
    "AH", "AH0", "AH1", "AH2",
    "AO", "AO0", "AO1", "AO2",
    "AW", "AW0", "AW1", "AW2",
    "AY", "AY0", "AY1", "AY2",
    "EH", "EH0", "EH1", "EH2",
    "ER", "ER0", "ER1", "ER2",
    "EY", "EY0", "EY1", "EY2",
    "IH", "IH0", "IH1", "IH2",
    "IY", "IY0", "IY1", "IY2",
    "OW", "OW0", "OW1", "OW2",
    "OY", "OY0", "OY1", "OY2",
    "UH", "UH0", "UH1", "UH2",
    "UW", "UW0", "UW1", "UW2",
]
non_stressable_phonetics = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG",
    "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
]
phonetics = stressable_phonetics + non_stressable_phonetics

# Numbers of each list
letters_num = len(letters)
stressable_phonetics_num = len(stressable_phonetics)
non_stressable_phonetics_num = len(non_stressable_phonetics)
phonetics_num = len(phonetics)

# Mappings
letter2id_dict = {k: letters.index(k) for k in letters}
id2letter_dict = {v: k for k, v in letter2id_dict.items()}
phonetic2id_dict = {k: phonetics.index(k) for k in phonetics}
id2phonetic_dict = {v: k for k, v in phonetic2id_dict.items()}
