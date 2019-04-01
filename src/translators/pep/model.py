from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout


class PredictionModel(object):
    def __init__(self):
        self._model = Sequential()
        self._model.add(Embedding(output_dim=32, input_dim=3800, input_length=380))
        self._model.add(Dropout(0.2))
        self._model.add(LSTM(32))
        self._model.add(Dense(units=256, activation='relu'))
        self._model.add(Dropout(0.2))
        self._model.add(Dense(units=1, activation='sigmoid'))

    @property
    def model(self):
        return self._model
