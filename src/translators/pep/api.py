# You should use me to visit this package from outer scope!
from translators.pep.constants import PHONETIC_DICT_PATH
from translators.pep.model import PredictionModel
from translators.pep.utils import load_pickle_dict


class PhoneticDictionary:

    def __init__(self):
        self.predict_model = PredictionModel()
        self._d = load_pickle_dict(PHONETIC_DICT_PATH)

    def __contains__(self, word):
        return word.upper() in self._d

    def __len__(self):
        return len(self._d)

    def __getitem__(self, word):
        return self._d[word.upper()]

    def __iter__(self):
        return iter(self._d)

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def values(self):
        return self._d.values()

    def lookup(self, word):
        entry = None
        if word.upper() in self._d:
            all_entries = self._d[word.upper()]
            if len(all_entries) != 0:
                entry = all_entries[0]
        return entry

    def predict(self, word):
        return self.predict_model.predict(word)

    def lookup_or_predict(self, word):
        result_from_dict = self.lookup(word)
        return self.predict(word) if not result_from_dict else result_from_dict
