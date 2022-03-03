import pandas as pd


class DataClass:

    def __init__(self):
        self.__X_train: pd.Series = None            #features for training
        self.__X_test: pd.Series = None             #features for evaluation
        self.__y_train: pd.Series = None            #labels for training
        self.__y_test: pd.Series = None             #labels for testing

    # Property decorators ----------------------------------------------------------------------------------------------
    @property
    def X_train(self) -> pd.Series:
        return self.__X_train

    @X_train.setter
    def X_train(self, value: pd.Series):
        self.__X_train = value

    @property
    def X_test(self) -> pd.Series:
        return self.__X_test

    @X_test.setter
    def X_test(self, value: pd.Series):
        self.__X_test = value

    @property
    def y_train(self) -> pd.Series:
        return self.__y_train

    @y_train.setter
    def y_train(self, value: pd.Series):
        self.__y_train = value

    @property
    def y_test(self) -> pd.Series:
        return self.__y_test

    @y_test.setter
    def y_test(self, value: pd.Series):
        self.__y_test = value

