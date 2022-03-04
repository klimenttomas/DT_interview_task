# the core of the code is based on key libraries: tensorflow, scikit-learn, pandas, numpy

from preprocessText.PreprocessText import PreprocessClass
from prepareForSVC.PrepareForSVCWorker import WorkerClass as PrepSVCWorker
from prepareForSVC.PrepareForSVCData import DataClass as PrepSVCData
from svc.SVCWorker import WorkerClass as SVCWorker
from prepareForANN.PrepareForANNWorker import WorkerClass as PrepANNWorker
from prepareForANN.PrepareForANNData import DataClass as PrepANNData
from ann.ANNWorker import WorkerClass as ANNWorker
import numpy as np
import pandas as pd


class Controller:

    def __init__(self, source_path: str):
        self.__source_path: str = source_path

    def do_it(self):
        # Object for preprocessing of the raw text
        prep: PreprocessClass = PreprocessClass()
        # Setting the path to source of raw data
        prep.source_path = self.__source_path
        # Setting if algorithm can remove blank lines after preprocessing, True - for training dataset, False - for
        # testing dataset
        prep.training_flag(True)
        # Preprocessing of the raw text
        prep.preprocess()


        # SVC - data objects and workers objects of SVC (classifier) ---------------------------------------------------
        worker_prep_svc: PrepSVCWorker = PrepSVCWorker(prep)
        # Prepare datasets
        worker_prep_svc.prepare_datasets()
        # Getting data object which holds input dataset for SVC
        data4svc: PrepSVCData = worker_prep_svc.SVCData

        # SVC worker object creation
        worker_svc: SVCWorker = SVCWorker(data4svc)
        # Creating and training the classifier
        worker_svc.classify()
        # Getting data object which holds trained classifier and result metrics
        results_svc = worker_svc.SVC_results

        # print metrics of the SVC
        print("SVC METRICS:\n")
        print(results_svc.metrics)


        # ANN - data objects and workers objects of ANN classifier------------------------------------------------------
        worker_prep_ann: PrepANNWorker = PrepANNWorker(prep)
        # Prepare datasets
        worker_prep_ann.prepare_datasets()
        # Getting data object which holds input dataset for ANN classifier
        data4ann: PrepANNData = worker_prep_ann.ANNData

        # ANN worker object creation
        worker_ann: ANNWorker = ANNWorker(data4ann)
        # Creating and training the classifier
        worker_ann.classify()
        # Getting data object which holds trained classifier and result metrics
        results_ann = worker_ann.ANN_results

        # print metrics of the ANN classifier
        print("ANN METRICS:\n")
        print(results_ann.metrics)

        # TESTING ------------------------------------------------------------------------------------------------------
        # For evaluation of the accuracy (and another metrics) of the classifier, final test data (test.csv, with no labels)
        # have to be preprocessed by the same way as trained data was.
        prep.source_path = "test.csv"
        # Calling this method with False parameters means, that blank lines after text preprocessing will NOT be removed
        # from final test file.
        prep.training_flag(False)
        prep.preprocess()
        df: pd.DataFrame = prep.prep_df

        # Predictions making on final testset by SVC -------------------------------------------------------------------
        svc_results: list = worker_svc.predict_on_test_set(df)
        svc_result_df = pd.DataFrame(svc_results, columns=['review_id', 'user_suggestion'])
        # saving predictions on final test file to csv
        svc_result_df.to_csv("sample_submission_svc.csv", sep=',', index=False)

        # Predictions making on final testset by ANN -------------------------------------------------------------------
        ann_results: list = worker_ann.predict_on_test_set(df)
        ann_result_df = pd.DataFrame(ann_results, columns=['review_id', 'user_suggestion'])
        # saving predictions on final test file to csv
        ann_result_df.to_csv("sample_submission_ann.csv", sep=',', index=False)

