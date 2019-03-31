from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout



class PredictionModel:

    def __init__(self):
        self.model = Sequential()
        self.model.add(Embedding(output_dim=32,
                            input_dim=3800,
                            input_length=380))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32))
        self.model.add(Dense(units=256,
                        activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1,
                        activation='sigmoid'))

    def summary(self):
        return self.model.summary()
