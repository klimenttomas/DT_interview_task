from sklearn.pipeline import Pipeline
from sklearn import metrics as sklearn_metrics


class DataClass:

    def __init__(self):
        self.__classifier: Pipeline = None                  # SVC (classifer with TF-IDF Vetorizer)
        self.__metrics: sklearn_metrics = None              # appropriate metrics: accuracy, recall, precession, f1-score

    # Property decorators ----------------------------------------------------------------------------------------------
    @property
    def classifier(self) -> Pipeline:
        return self.__classifier

    @classifier.setter
    def classifier(self, value: Pipeline):
        self.__classifier = value

    @property
    def metrics(self) -> sklearn_metrics:
        return self.__metrics

    @metrics.setter
    def metrics(self, value: sklearn_metrics):
        self.__metrics = value



