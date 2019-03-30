import csv
import pickle

from translators.pep.constants import CMU_DICT_PATH, CMU_SYMBOLS_PATH, PHONETIC_DICT_PATH, PHONETIC_SYMBOLS_PATH, \
    ABBREV_DICT_PATH, QTY_ABBREV_DICT_PATH, ABBREV_CSV_PATH, QTY_ABBREV_CSV_PATH


class PhoneticDictionaryGenerator:

    def __init__(self, skip_word_func):
        self.should_skip_word = skip_word_func
        self.phonetic_dict = None
        self.symbol_list = None

    @staticmethod
    def _is_alternate_pho_spelling(word):
        # No word has > 9 alternate pronunciations
        return word[-1] == ')' and word[-3] == '(' and word[-2].isdigit()

    def create_dict_from_cmu(self):
        phonetic_dict = {}
        with open(CMU_DICT_PATH, encoding="ISO-8859-1") as cmu_dict:
            for line in cmu_dict:

                # Skip commented lines
                if line[0:3] == ';;;':
                    continue

                word, phonetic = line.strip().split('  ')

                # Alternate pronounciations are formatted: "WORD(#)  F AH0 N EH1 T IH0 K"
                # We don't want to the "(#)" considered as part of the word
                if self._is_alternate_pho_spelling(word):
                    word = word[:word.find('(')]

                if self.should_skip_word(word):
                    continue

                if word not in phonetic_dict:
                    phonetic_dict[word] = []
                phonetic_dict[word].append(phonetic)

        return phonetic_dict

    def generate_phonetic_dict(self, save_path):
        phonetic_dict = self.create_dict_from_cmu()
        self._save(save_path, phonetic_dict)

    @staticmethod
    def create_symbol_list_from_cmu():
        phone_list = []
        with open(CMU_SYMBOLS_PATH) as file:
            for line in file:
                phone_list.append(line.strip())
        return phone_list

    def generate_symbol_list(self, save_path):
        symbol_list = self.create_symbol_list_from_cmu()
        self._save(save_path, symbol_list)

    @staticmethod
    def _save(save_path, data):
        with open(save_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def csv_to_pkl_dict(csv_path, pkl_dict_path):
    with open(csv_path, mode='r') as csv_handle:
        reader = csv.reader(csv_handle)
        pkl_dict = {}
        for row in reader:
            row_key = row[0].strip()
            row_vals = [val.strip() for val in row[1:]]
            if len(row_vals) == 1:
                pkl_dict[row_key] = row_vals[0]
            else:
                pkl_dict[row_key] = row_vals
    with open(pkl_dict_path, 'wb') as pkl_handle:
        pickle.dump(pkl_dict, pkl_handle, protocol=pickle.HIGHEST_PROTOCOL)


def should_skip_lookup_dict(word):
    return not word[0].isalpha()

if __name__=='__main__':
    choice = input('This will rewrite *.pkl in output/! Proceed?[y/N]')
    if choice is not 'y' or 'Y':
        exit(0)
    gen = PhoneticDictionaryGenerator(should_skip_lookup_dict)
    gen.generate_phonetic_dict(PHONETIC_DICT_PATH)
    gen.generate_symbol_list(PHONETIC_SYMBOLS_PATH)
    csv_to_pkl_dict(ABBREV_CSV_PATH, ABBREV_DICT_PATH)
    csv_to_pkl_dict(QTY_ABBREV_CSV_PATH, QTY_ABBREV_DICT_PATH)
    print('Dictionaries Created Successfully!')
