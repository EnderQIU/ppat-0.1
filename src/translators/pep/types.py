from string import ascii_uppercase, digits

import numpy as np

from translators.pep.constants import MAX_WORD_LEN, MIN_WORD_LEN, MAX_PRONUNCIATION_LEN, MIN_PRONUNCIATION_LEN

remove_digits = str.maketrans(digits)
# A-Z
letters = list(ascii_uppercase)
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


class Word(object):
    def __init__(self, text):
        assert isinstance(text, str) and text.isalpha() and MIN_WORD_LEN <= len(text) <= MAX_WORD_LEN
        self._text = text
        self._one_hot = np.zeros([MAX_WORD_LEN, letters_num], dtype=int, order='C')
        r = 0
        for s in text:
            self._one_hot[r][letter2id_dict[s]] = 1
            r += 1

    @property
    def text(self):
        return self._text

    @property
    def one_hot(self):
        return self._one_hot


class Pronunciation(object):
    def __init__(self, src=None):
        if isinstance(src, str):
            # init from str
            assert MIN_PRONUNCIATION_LEN <= len(src.split(' ')) <= MAX_PRONUNCIATION_LEN

            self._text = src
            self._one_hot = np.zeros([MAX_PRONUNCIATION_LEN, phonetics_num], dtype=int, order='C')
            r = 0
            for s in src:
                self._one_hot[r][phonetic2id_dict[s]] = 1
                r += 1
        else:
            # init from one-hot vector
            assert src.shape == (MAX_PRONUNCIATION_LEN, phonetics_num)

            self._one_hot = src
            self._text = ''
            for r in src:
                index = 0
                for i in r:
                    if i == 1:
                        break
                    index += 1
                self._text += id2phonetic_dict[index] + ' '
            self._text.rstrip()  # remove the last ' '

        self._syllable_text = self._text.translate(remove_digits)
        self._syllable_one_hot = np.zeros([MAX_PRONUNCIATION_LEN, phonetics_num], dtype=int, order='C')
        r = 0
        for s in src:
            index = phonetic2id_dict[s]
            if index < stressable_phonetics_num:
                index = index // 4
            self._syllable_one_hot[r][index] = 1
            r += 1

    @property
    def text(self):
        return self._text

    @property
    def one_hot(self):
        return self._one_hot

    def equal(self, other):
        """
        Compare self to other.
        other could be a string or one-hot vector.
        :param other:
        :return: True for equivalent, False for totally different
        """
        if isinstance(other, str):
            return self._text == other
        else:
            return self._one_hot == other

    def syllable_equal(self, other):
        """
        Equivalent in syllables, ignoring the stresses
        :param other:
        :return:
        """
        if isinstance(other, str):
            return self._syllable_text == other.translate(remove_digits)
        else:
            syllable_other = np.zeros([MAX_PRONUNCIATION_LEN, phonetics_num], dtype=int, order='C')
            row_count = 0
            for r in other:
                index = 0
                for i in r:
                    if i == 1:
                        break
                    index += 1
                if index < stressable_phonetics_num:
                    index = index // 4
                syllable_other[row_count][index] = 1
            return self._syllable_one_hot == syllable_other
