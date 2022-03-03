from prepareForANN.PrepareForANNData import DataClass as ANNInputClass
from .ANNData import DataClass as ANNOutputClass
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
from sklearn import metrics as sklearn_metrics
import numpy as np
import pandas as pd


class WorkerClass:

    def __init__(self, data: ANNInputClass):
        self.__data: ANNInputClass = data                               # Inpute data for classifier
        self.__ANN_results: ANNOutputClass = ANNOutputClass()           # Output data - trained classifier and metrics
        self.__classifier: Sequential = None                            # ANN Classifier
        self.__predictions = None                                       # Predictions from test dataset (created from train dataset)
        #self.__embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
        self.__early_stop: EarlyStopping = None                         # Stop criteria for ANN as a protection from overtraining
        self.__history = None                                           # Data from ANN training

    # Property decorators ----------------------------------------------------------------------------------------------
    @property
    def ANN_results(self) -> ANNOutputClass:
        return self.__ANN_results

    def classify(self):
        self.__create_classifier()
        self.__compile_classifier()
        self.__create_early_stop()
        self.__fit()
        self.__do_predictions()
        self.__evaluate_metrics()
        self.__safe_classifier()

    # ANN creating
    def __create_classifier(self):
        # hub_layer = hub.KerasLayer(self.__embedding, input_shape=[], dtype=tf.string, trainable=True)
        VOCAB_SIZE = 3000
        encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        encoder.adapt(self.__data.X_train)

        self.__classifier = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1)
        ])

    def __compile_classifier(self):
        self.__classifier.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=["accuracy"])

    def __create_early_stop(self):
        self.__early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=2)

    # Training the ANN
    def __fit(self):
        self.__history = self.__classifier.fit(x=self.__data.X_train, y=self.__data.y_train, epochs=10, validation_data=(self.__data.X_test, self.__data.y_test), callbacks=[self.__early_stop])

    # Predictions making on the test set (created from training set)
    def __do_predictions(self):
        self.__predictions = self.__classifier.predict(x=np.array(self.__data.X_test))

        # Output of the ANN has to be mapped to 0 and 1
        for i in range(len(self.__predictions)):
            if self.__predictions[i] >= 0.0:
                self.__predictions[i] = 1
            else:
                self.__predictions[i] = 0

    # Metrics calculation
    def __evaluate_metrics(self):
        self.__ANN_results.metrics = sklearn_metrics.classification_report(self.__data.y_test, self.__predictions)

    def __safe_classifier(self):
        self.__ANN_results.classifier = self.__classifier

    # PREDICTIONS MAKING ON FINAL TEST DATA (test.csv)
    def predict_on_test_set(self, pd_test: pd.DataFrame) -> list:
        results: list = []

        #for i in range(10):
        for i in range(len(pd_test)):
            id = str(pd_test.loc[i, "review_id"])
            review = pd_test.loc[i, "new_feature"]
            result = self.__classifier.predict(np.array([review])).item(0)
            # Output of the ANN has to be mapped to 0 and 1
            if result >= 0.0:
                result = 1
            else:
                result = 0
            results.append((id, str(result)))

        return results
