from preprocessText.PreprocessText import PreprocessClass
from .PrepareForSVCData import DataClass as SVCInputData
from sklearn.model_selection import train_test_split as tts


class WorkerClass:

    def __init__(self, data: PreprocessClass, test_size: float = 0.15):
        self.__data: PreprocessClass = data                     # Preprocessed text data
        self.__test_size: float = test_size                     # Test data size definition (as a proportion from original data)
        self.__SVCData: SVCInputData = SVCInputData()           # Split datasets - input data for classificator

    # Property decorators ----------------------------------------------------------------------------------------------
    @property
    def SVCData(self) -> SVCInputData:
        return self.__SVCData

    @SVCData.setter
    def SVCData(self, value: SVCInputData):
        self.__SVCData = value

    # Method which prepares preprocessed text data for 4 datasets: feature train, feauture test, label train, label test
    # The division into a training and test dataset from the originally supplied training dataset (train.csv) is
    # important for classifier evaluation. The final test data (test.csv - as a part of the interview task) will be
    # tested later
    def prepare_datasets(self):
        X = self.__data.prep_df["new_feature"]
        y = self.__data.prep_df["user_suggestion"]
        self.__SVCData.X_train, self.__SVCData.X_test, self.__SVCData.y_train, self.__SVCData.y_test = tts(X, y, test_size=self.__test_size, random_state=12)

