import re
import os
import pickle

import numpy as np

from translators.pep.constants import ALLOWED_CHARS
from translators.pep.constants import PHONETIC_SYMBOLS_PATH
from translators.pep.constants import START_PHONE_SYM, END_PHONE_SYM, MAX_CHAR_SEQ_LEN


def safe_split(phrase):

    def substring_indexes(substring, string):
        last_found = -1  # Begin at -1 so the next position to search from is 0
        while True:
            # Find next index of substring, by starting after its last known position
            last_found = string.find(substring, last_found + 1)
            if last_found == -1:
                break  # All occurrences have been found
            yield last_found

    def replace_periods_with_spaces(_phrase):
        for period_idx in list(substring_indexes('.', _phrase)):
            if not (_phrase[period_idx-1].isdigit() and _phrase[period_idx+1].isdigit()):
                char_list = list(_phrase)
                char_list[period_idx] = ' '
                _phrase = ''.join(char_list)
        return _phrase

    phrase = replace_periods_with_spaces(phrase)
    return re.split(r'[\s\-_—]+', phrase.strip())


def sanitize(word):
    allowed_chars = re.escape(''.join(ALLOWED_CHARS))
    return re.sub(r"[^"+allowed_chars+"]", '', word.upper())


def is_number(word):
    return len(word) != 0 and not bool(re.search(r'[^\d\\.,]', word))


def contains_digits(word):
    return len(word) != 0 and bool(re.search(r'(\d+)', word))


def space_pad_regex(input_string, regex):
    return re.sub(regex, r' \1 ', input_string).strip()


def count_phonemes_with_emphasis(phonetic_sp):

    def phone_has_emphasis(_phone):
        if len(_phone) == 0:
            return False
        return _phone.strip()[-1].isdigit()

    count = 0
    for phone in phonetic_sp.split():
        if phone_has_emphasis(phone):
            count += 1
    return count


def load_pickle_dict(dict_path):
    if not os.path.isfile(dict_path):
        raise FileNotFoundError('The dictionary file is missing. Generate it by running '
                                '`python dev/generate_dictionaries.py`')
    with open(dict_path, 'rb') as handle:
        return pickle.load(handle)


class PredictionModelUtils:

    def __init__(self):
        self.chars = self._char_list()
        self.phones = self._phone_list()
        self.char_token_count = len(self.chars)
        self.phone_token_count = len(self.phones)
        self.char_to_id, self.id_to_char = self._id_mappings_from_list(self.chars)
        self.phone_to_id, self.id_to_phone = self._id_mappings_from_list(self.phones)

    @staticmethod
    def _char_list():
        return [''] + ALLOWED_CHARS

    @staticmethod
    def _phone_list():
        edge_symbols = [START_PHONE_SYM, END_PHONE_SYM]
        cmu_symbols = load_pickle_dict(PHONETIC_SYMBOLS_PATH)
        return [''] + edge_symbols + cmu_symbols

    @staticmethod
    def _id_mappings_from_list(str_list):
        str_to_id = {s: i for i, s in enumerate(str_list)}
        id_to_str = {i: s for i, s in enumerate(str_list)}
        return str_to_id, id_to_str

    def word_to_char_ids(self, word):
        word_vec = np.zeros(MAX_CHAR_SEQ_LEN)
        t = 0
        for char in word:
            if char in self.char_to_id:
                word_vec[t] = self.char_to_id[char]
                t += 1
            if t >= MAX_CHAR_SEQ_LEN:
                break
        return np.array([word_vec])
