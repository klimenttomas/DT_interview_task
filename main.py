# Created by Tomas Kliment, with virtual interpreter based on Python 3.12
# The goal of this program is to classify game's user suggestion based on available data as:
# - user review (of the game)
# - title of the game
# Other data as year and ID was not considered for Machine Learning (ML) classifiers
# In this program relevant text data was preprocessed and 2 independent ML classifiers was created and trained:
# - SVC (support vector classifier)
# - ANN (artificial neural network) classifier
# After training, delivered test set (named test.csv) was used for both classificators evaluation (accuracy, f1-score, etc).
# This program is divided to several packages:
# - controller: contains top level objects which manipulate with data and appropriate workers
# - preprocessText: this package is responsible for raw text preprocessing
# - prepareForSVC: preparation preprocessed text data for SVC - data splitting
# - prepareForANN: preparation preprocessed text data for ANN Classifier - data splitting
# - svc: bulding and training SVC and appropriate metrics evaluation
# - ann: bulding and training ANN classifier and appropriate metrics evaluation
# More information about code functionality is situated in the specific packages

from controller.MainController import Controller


if __name__ == "__main__":

    ctr: Controller = Controller("train.csv")
    ctr.do_it()