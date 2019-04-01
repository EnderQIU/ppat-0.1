import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, EarlyStopping

from translators.pep.constants import PREDICTION_MODEL_WEIGHTS_PATH
from translators.pep.data_provider import PronunciationDataProvider
from translators.pep.model import PredictionModel
from translators.pep.types import Pronunciation


class PredictionModelTrainer:

    evaluate_size = 0.2

    def __init__(self, _model):
        self.data_provider = PronunciationDataProvider()
        self.model = _model
        self.train_history = None
        self.x_train = None
        self.y_train = None

    def train(self,
              loss='categorical_crossentropy',
              optimizer='adam',
              monitor='val_loss',
              patience=3,
              batch_size=100,
              epochs=100,
              check_point_verbose=1,
              fit_verbose=2,
              validation_split=0.2,
              ):
        self.model.model.compile(loss=loss, optimizer=optimizer)

        self.x_train, self.y_train = self.data_provider.train_dict.kvnarray
        checkpoint = ModelCheckpoint(filepath=PREDICTION_MODEL_WEIGHTS_PATH,
                                     verbose=check_point_verbose,
                                     save_best_only=True)
        stopper = EarlyStopping(monitor=monitor, patience=patience)
        callbacks = [checkpoint, stopper]
        self.train_history = self.model.model.fit(self.x_train,
                                                  self.y_train,
                                                  batch_size=batch_size,
                                                  epochs=epochs,
                                                  verbose=fit_verbose,
                                                  validation_split=validation_split,
                                                  callbacks=callbacks,
                                                  )

    def graph_train_history(self, train, validation):
        if not self.train_history:
            print('Model not trained! Call train() first!')
        plt.plot(self.train_history.history[train])
        plt.plot(self.train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def evaluate(self):
        correct_syllable_counts = 0
        perfect_predictions = 0
        index = 0
        for i in tqdm(range(len(self.data_provider.evaluation_dict))):
            word = self.data_provider.evaluation_dict.keys()[i]
            predict_pronunciation = Pronunciation(self.model.model.predict(word.one_hot))
            real_pronunciation = self.data_provider.evaluation_dict[self.data_provider.evaluation_dict.keys()[i]]
            if predict_pronunciation.equal(real_pronunciation):
                correct_syllable_counts += 1
                perfect_predictions += 1
            elif predict_pronunciation.syllable_equal(real_pronunciation):
                correct_syllable_counts += 1
            index += 1

        syllable_acc = correct_syllable_counts / self.data_provider.evaluation_dict.__len__()
        perfect_acc = perfect_predictions / self.data_provider.evaluation_dict.__len__()

        print('Syllable Accuracy: {}%'.format(round(syllable_acc * 100, 1)))
        print('Perfect Accuracy: {}%'.format(round(perfect_acc * 100, 1)))


if __name__ == '__main__':
    choice = input('This will rewrite the current model and take up to 2 hours(i7-6700 with GTX1070)! Proceed?[y/N]')
    if choice not in ['y', 'Y']:
        exit(0)
    # Training
    model = PredictionModel()
    model.model.summary()
    model_trainer = PredictionModelTrainer(model)
    model_trainer.train()
    model_trainer.graph_train_history('acc','val_acc')

    # Evaluation
    model_trainer.evaluate()
