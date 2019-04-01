from random import sample

import numpy as np

from translators.pep.constants import MAX_WORD_LEN, MIN_WORD_LEN, MAX_PRONUNCIATION_LEN, MIN_PRONUNCIATION_LEN, \
    letters_num, letter2id_dict, phonetics_num, phonetic2id_dict, id2phonetic_dict, remove_digits, \
    stressable_phonetics_num


def is_pure_english_alpha(text):
    return all(64 < ord(c) < 91 or 96 < ord(c) < 123 for c in text)


class Word(object):
    def __init__(self, text):
        assert isinstance(text, str) and is_pure_english_alpha(text) and MIN_WORD_LEN <= len(text) <= MAX_WORD_LEN
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

    def __hash__(self):
        return hash(self._text)

    def __len__(self):
        return len(self._text)


class Pronunciation(object):
    def __init__(self, src=None):
        if isinstance(src, str):
            # init from str
            assert MIN_PRONUNCIATION_LEN <= len(src.split(' ')) <= MAX_PRONUNCIATION_LEN

            self._text = src
            self._one_hot = np.zeros([MAX_PRONUNCIATION_LEN, phonetics_num], dtype=int, order='C')
            r = 0
            for s in src.split(' '):
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
        for s in self._syllable_text.split(' '):
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

    @property
    def syllable_one_hot(self):
        return self._syllable_one_hot

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
        assert isinstance(other, Pronunciation)

        return self._syllable_one_hot == other.syllable_one_hot

    def __hash__(self):
        return hash(self._text)

    def __len__(self):
        return len(self._text)


class SampleDict(dict):
    _kvnarray = None

    def __setitem__(self, key, value):
        if self._kvnarray:
            raise TypeError('SampleDict does nto support item assignment once called kvarray()')
        else:
            return super().__setitem__(key, value)

    def sample(self, scale):
        """
        Randomly pick scale*100 % items
        :param scale:
        :return:
        """
        assert 0 < scale < 1

        return SampleDict({k: self[k] for k in sample(list(self.keys()), int(scale*100))})

    def slice(self, scale):
        """
        Slice items into two parts at position of scale
        :param scale:
        :return:
        """
        assert 0 < scale < 1

        border = int(scale * self.__len__())
        count = 0
        ahead = SampleDict()
        rest = SampleDict()
        for k in self.keys():
            if count < border:
                ahead[k] = self[k]
            else:
                rest[k] = self[k]
        return ahead, rest

    @property
    def kvnarray(self):
        if self._kvnarray:
            return self._kvnarray

        assert all(isinstance(k, Word) for k in self.keys()) \
            and all(isinstance(v, list) for v in self.values())

        k_list = []
        v_list = []
        for k in self.keys():
            for v in self[k]:
                k_list.append(k.one_hot)
                v_list.append(v.one_hot)
        self._kvnarray = (np.array(k_list), np.array(v_list))
        return self._kvnarray
