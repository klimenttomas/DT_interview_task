from prepareForSVC.PrepareForSVCData import DataClass as SVCInputClass
from .SVCData import DataClass as SVCOutputClass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics as sklearn_metrics
import pandas as pd


class WorkerClass:

    def __init__(self, data: SVCInputClass):
        self.__data: SVCInputClass = data                       # Inpute data for classifier
        self.__SVC_results: SVCOutputClass = SVCOutputClass()   # Output data - trained classifier and metrics
        self.__classifier: Pipeline = None                      # Classifier with TF-IDF Vectorizer
        self.__predictions = None                               # Predictions from test dataset (created from train dataset)

    # Property decorators ----------------------------------------------------------------------------------------------
    @property
    def SVC_results(self) -> SVCOutputClass:
        return self.__SVC_results


    def classify(self):
        self.__create_classifier()
        self.__feed_and_fit()
        self.__do_predictions()
        self.__evaluate_metrics()
        self.__safe_classifier()

    # Classifier with TF-IDF Vetorizer creation
    def __create_classifier(self):
        self.__classifier = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC()),])

    # Training the SVC
    def __feed_and_fit(self):
        self.__classifier.fit(self.__data.X_train, self.__data.y_train)

    # Predictions making on the test set (created from training set)
    def __do_predictions(self):
        self.__predictions = self.__classifier.predict(self.__data.X_test)

    # Metrics calculation
    def __evaluate_metrics(self):
        self.__SVC_results.metrics = sklearn_metrics.classification_report(self.__data.y_test, self.__predictions)

    def __safe_classifier(self):
        self.__SVC_results.classifier = self.__classifier

    def predict(self, text: str or list):
        result = None
        if type(text) == str:
            result = self.__classifier.predict([text])

        else:
            result = self.__classifier.predict(text)

    # PREDICTIONS MAKING ON FINAL TEST DATA (test.csv)
    def predict_on_test_set(self, pd_test: pd.DataFrame) -> list:
        results: list = []

        for i in range(len(pd_test)):
            id = str(pd_test.loc[i, "review_id"])
            review = pd_test.loc[i, "new_feature"]
            result = str(self.__classifier.predict([review])[0])
            results.append((id, result))

        return results


