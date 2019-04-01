from translators.pep.constants import CMU_DICT_PATH, MIN_WORD_LEN, MAX_WORD_LEN, DATA_LIMIT_SCALE, TRAIN_SCALE, \
    MIN_PRONUNCIATION_LEN, MAX_PRONUNCIATION_LEN
from translators.pep.types import Word, Pronunciation, SampleDict, is_pure_english_alpha


class PronunciationDataProvider:

    def __init__(self):
        self.load_dict = SampleDict()
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

                if is_pure_english_alpha(left) and MIN_WORD_LEN < len(left) < MAX_WORD_LEN \
                        and MIN_PRONUNCIATION_LEN < len(phonetics.split(' ')) < MAX_PRONUNCIATION_LEN:
                    word = Word(left)
                    pronunciation = Pronunciation(phonetics)
                    if word in self.load_dict.keys():
                        self.load_dict[word].append(pronunciation)
                    else:
                        self.load_dict[word] = [pronunciation]

        self.load_dict = self.load_dict.sample(DATA_LIMIT_SCALE)
        self.train_dict, self.evaluation_dict = self.load_dict.slice(TRAIN_SCALE)
