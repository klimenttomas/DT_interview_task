import pandas as pd
import numpy as np


class DataClass:

    def __init__(self):
        self.__X_train: np.ndarray = None            #features for training
        self.__X_test: np.ndarray = None             #features for evaluation
        self.__y_train: np.ndarray = None            #labels for training
        self.__y_test: np.ndarray = None             #labels for testing

    # Property decorators ----------------------------------------------------------------------------------------------
    @property
    def X_train(self) -> np.ndarray:
        return self.__X_train

    @X_train.setter
    def X_train(self, value: np.ndarray):
        self.__X_train = value

    @property
    def X_test(self) -> np.ndarray:
        return self.__X_test

    @X_test.setter
    def X_test(self, value: np.ndarray):
        self.__X_test = value

    @property
    def y_train(self) -> np.ndarray:
        return self.__y_train

    @y_train.setter
    def y_train(self, value: np.ndarray):
        self.__y_train = value

    @property
    def y_test(self) -> np.ndarray:
        return self.__y_test

    @y_test.setter
    def y_test(self, value: np.ndarray):
        self.__y_test = value

