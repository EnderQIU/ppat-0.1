from random import sample
from string import digits, ascii_uppercase

import numpy as np
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer

from translators.pep.constants import CMU_DICT_PATH, MIN_WORD_LEN, MAX_WORD_LEN, DATA_LIMIT_SCALE, TRAIN_SCALE

remove_digits = str.maketrans('', '', digits)


class PronunciationDataProvider:

    def __init__(self):
        # ["A", "B", ..., "Z"]
        self.letters = list(ascii_uppercase)
        # All phonetics
        self.stressable_phonetics = [
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
        self.stressable_phonetics_len = len(self.stressable_phonetics)
        self.non_stressable_phonetics = [
            "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG",
            "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
        ]
        self.phonetics = self.stressable_phonetics + self.non_stressable_phonetics
        # Dictionary for predict and train, purely alphabetical
        self.load_dict = {}
        self.max_word_len = 0
        self.max_phonetics_num = 0

        with open(CMU_DICT_PATH, encoding="ISO-8859-1") as f:
            for line in f:
                # Skip commented lines
                if line[0:3] == ';;;':
                    continue
                left, phonetics = line.strip().split('  ')
                assert isinstance(left, str) and isinstance(phonetics, str)
                # Alternate pronounciations are formatted: "WORD(#)  F AH0 N EH1 T IH0 K"
                # We don't want to the "(#)" considered as part of the word
                if left.endswith(')'):
                    left = left[:left.find('(')]

                if left.isalpha() and MIN_WORD_LEN < len(left) < MAX_WORD_LEN:
                    if left in self.load_dict.keys():
                        self.load_dict[left].append(phonetics)
                    else:
                        self.load_dict[left] = [phonetics]
                    # Update the max_word_len and max_phonetics_num
                    if len(left) > self.max_word_len:
                        self.max_word_len = len(left)
                    if len(phonetics.split(' ')) > self.max_phonetics_num:
                        self.max_phonetics_num = len(phonetics.split(' '))
        self.load_dict_length = int(DATA_LIMIT_SCALE * len(self.load_dict.keys()))
        self.load_dict = {k: self.load_dict[k] for k in sample(list(self.load_dict.keys()), self.load_dict_length)}
        self.train_dict = {}
        self.evaluation_dict = {}
        border = int(TRAIN_SCALE * self.load_dict_length)
        count = 0
        for k in self.load_dict:
            if count < border:
                self.train_dict[k] = self.load_dict[k]
            else:
                self.evaluation_dict[k] = self.load_dict[k]
            count += 1
        self.train_dict_length = border
        self.evaluation_dict_length = self.load_dict_length - border

        self.train_generated = False
        self.x_train = []
        self.y_train = []

        self.evaluate_generated = False
        self.x_evaluate = []
        self.y_evaluate = []

        self.xy_dict = {}

        self.tokenizers_generated = False
        self.word_tokenizer = None
        self.words_token_num = 0
        self.phonetics_tokenizer = None
        self.phonetics_token_num = 0

    def word_to_sequence(self, word):
        assert isinstance(word, str)

        return sequence.pad_sequences(
            self.word_tokenizer.texts_to_sequences(' '.join(word)),
            maxlen=self.max_word_len
        )

    def pronunciation_to_sequence(self, pronunciation):
        assert isinstance(pronunciation, str)

        return sequence.pad_sequences(
            self.phonetics_tokenizer.texts_to_sequences(pronunciation),
            maxlen=self.max_phonetics_num),

    @staticmethod
    def pronunciation_to_syllables(pronunciation):
        """
        Remove stress integers in pronunciations
        :param pronunciation:
        :return:
        """
        assert isinstance(pronunciation, str)

        return pronunciation.translate(remove_digits)

    @staticmethod
    def sequence_to_one_hot_matrix(seq, num):
        assert isinstance(seq, list) and all([isinstance(i, int) for i in seq])
        assert isinstance(num, int)

        one_hot_matrix = []
        for s in seq:
            one_hot_vector = np.zeros(num)
            if s != 0:
                one_hot_vector[s - 1] = 1
            one_hot_matrix.append(one_hot_vector)
        return one_hot_matrix

    def _build_tokenizers(self):
        if self.tokenizers_generated:
            return
        self.word_tokenizer = Tokenizer(lower=False, char_level=True)
        self.word_tokenizer.fit_on_texts(self.letters)
        self.words_token_num = len(self.word_tokenizer.word_index)
        self.phonetics_tokenizer = Tokenizer(lower=False)
        self.phonetics_tokenizer.fit_on_texts(' '.join(self.phonetics))
        self.phonetics_token_num = len(self.word_tokenizer.word_index)
        self.tokenizers_generated = True

    def get_train(self):
        if self.train_generated:
            return self.x_train, self.y_train
        else:
            self._build_tokenizers()
            for word in self.train_dict.keys():
                pronunciations = self.train_dict[word]
                for pronunciation in pronunciations:
                    self.x_train.append(
                        self.sequence_to_one_hot_matrix(
                            self.word_to_sequence(word),
                            self.words_token_num
                        )
                    )
                    self.y_train.append(
                        self.sequence_to_one_hot_matrix(
                            self.pronunciation_to_sequence(pronunciation),
                            self.phonetics_token_num
                        )
                    )
            assert len(self.x_train) == len(self.y_train)
            self.train_generated = True
            return self.x_train, self.y_train

    def get_evaluate(self):
        if self.evaluate_generated:
            return self.x_evaluate, self.y_evaluate
        else:
            self._build_tokenizers()
            for word in self.evaluation_dict.keys():
                pronunciations = self.evaluation_dict[word]
                for pronunciation in pronunciations:
                    self.x_evaluate.append(
                        self.sequence_to_one_hot_matrix(
                            self.word_to_sequence(word),
                            self.words_token_num
                        )
                    )
                    self.y_evaluate.append(
                        self.sequence_to_one_hot_matrix(
                            self.pronunciation_to_sequence(pronunciation),
                            self.phonetics_token_num
                        )
                    )
            assert len(self.x_evaluate) == len(self.y_evaluate)
            self.evaluate_generated = True
            return self.x_evaluate, self.y_evaluate

    def _build_xy_dict(self):
        for i in range(len(self.x_train)):
            x = self.x_train[i]
            y = self.y_train[i]
            if x in self.xy_dict.keys():
                self.xy_dict[x].append(y)
            else:
                self.xy_dict[x] = []
        for i in range(len(self.x_evaluate)):
            x = self.x_evaluate[i]
            y = self.y_evaluate[i]
            if x in self.xy_dict.keys():
                self.xy_dict[x].append(y)
            else:
                self.xy_dict[x] = []

    @staticmethod
    def _one_hot_to_sequence_index(one_hot):
        """
        Note the sequence index starts from 0,
        -1 represents an all zero array.
        :param one_hot:
        :return:
        """
        index = -1
        for i in one_hot:
            if i == 1.0:
                return index + 1
        return -1

    def _get_non_stress_sequence_from_matrix(self, stressed_one_hot_matrix):
        _sequence = []
        for one_hot in stressed_one_hot_matrix:
            index = self._one_hot_to_sequence_index(one_hot)
            if -1 < index < self.stressable_phonetics_len:
                index = index // 4
            _sequence.append(index)
        return _sequence

    def is_correct(self, word_one_hot_matrix, pronunciation_one_hot_matrix):
        if not self.xy_dict:
            self._build_xy_dict()
        return pronunciation_one_hot_matrix in self.xy_dict[word_one_hot_matrix]

    def is_syllable_correct(self, word_one_hot_matrix, pronunciation_one_hot_matrix):
        if not self.xy_dict:
            self._build_xy_dict()

        if self._get_non_stress_sequence_from_matrix(pronunciation_one_hot_matrix) in\
                [self._get_non_stress_sequence_from_matrix(m) for m in self.xy_dict[word_one_hot_matrix]]:
            return True
        else:
            return False
