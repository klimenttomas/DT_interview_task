import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import copy


class PreprocessClass:

    def __init__(self):
        self.__source_path: str = ""                # path to source
        self.__orig_df: pd.DataFrame = None         # original dataframe from source
        self.__prep_df: pd.DataFrame = None         # preprocessed dataframe
        # nltk data downloading
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('omw-1.4')
        self.__stop = stopwords.words('english')    # stop words creating
        self.__wnl = WordNetLemmatizer()            # object for lemmatization
        self.__drop_lines: bool = None              # flag, which indicates if blank lines has to be removed

    # Property decorators ----------------------------------------------------------------------------------------------
    @property
    def source_path(self) -> str:
        return self.__source_path

    @source_path.setter
    def source_path(self, value: str):
        self.__source_path: str = value
        self.__orig_df = None
        self.__prep_df = None

    @property
    def orig_df(self) -> pd.DataFrame:
        return self.__orig_df

    @orig_df.setter
    def orig_df(self, value: pd.DataFrame):
        self.__orig_df = value

    @property
    def prep_df(self) -> pd.DataFrame:
        return self.__prep_df

    @prep_df.setter
    def prep_df(self, value: pd.DataFrame):
        self.__prep_df = value

    # Flag if is it training or testing data - in testing data no row will be removed
    def training_flag(self, drop: bool):
        self.__drop_lines = drop

    # Reading from source and dataframe creating
    def __read_from_source(self):
        self.__orig_df = pd.read_csv(self.__source_path)

    # Preprocessing the raw text data - functionality of the used methods is desribed below
    def preprocess(self):
        self.__read_from_source()
        self.__prep_df = copy.deepcopy(self.__orig_df)
        self.__prep_df["title"] = self.__prep_df["title"].apply(self.__remove_blanks)

        self.__prep_df["user_review"] = self.__prep_df["user_review"].apply(self.__text2lower)
        self.__prep_df["user_review"] = self.__prep_df["user_review"].apply(self.__remove_wasteful_words)
        self.__prep_df["user_review"] = self.__prep_df["user_review"].apply(self.__remove_long_words)
        self.__prep_df["user_review"] = self.__prep_df["user_review"].apply(self.__remove_punctuation)
        self.__prep_df["user_review"] = self.__prep_df["user_review"].apply(self.__remove_numbers)
        self.__prep_df["user_review"] = self.__prep_df["user_review"].apply(self.__remove_stop_words)
        self.__prep_df["user_review"] = self.__prep_df["user_review"].apply(self.__lematization)

        if self.__drop_lines:
            self.__prep_df = self.__drop_no_information(self.__prep_df, "user_review")

        # User reviews text information is extended by title of the game. Is a assumption, that some games can be better
        # than the others in general and have less negative user suggestions.
        self.__prep_df["new_feature"] = self.__prep_df["title"] + ": " + self.__prep_df["user_review"]

    # Methods for text preprocessing -----------------------------------------------------------------------------------
    # Removing the stop words from text - they don't hold the information
    def __remove_stop_words(self, text: str) -> str:
        words_list = text.split()
        result_words = [word for word in words_list if word.lower() not in self.__stop]
        new_text = ' '.join(result_words)
        return new_text

    # Lemmatization of the text
    def __lematization(self, text: str) -> str:
        words_list = text.split()
        result_words = [self.__wnl.lemmatize(w) for w in words_list]
        new_text = ' '.join(result_words)
        return new_text

    # Converting to lower case
    @staticmethod
    def __text2lower(text: str) -> str:
        return text.lower()

    # Removing the punctuation
    @staticmethod
    def __remove_punctuation(text: str) -> str:
        new_text = re.sub(r'[^\w\s]', "", text)
        return new_text

    # Removing inappropriate long words, which can be considered as misspelling
    @staticmethod
    def __remove_long_words(text: str) -> str:
        words_list = text.split()
        result_words = [word for word in words_list if len(word) < 15]
        new_text = ' '.join(result_words)
        return new_text

    # Removing numbers from text
    @staticmethod
    def __remove_numbers(text: str) -> str:
        new_text = ''.join([letter for letter in text if not letter.isdigit()])
        return new_text

    # Some words in the training dataset don't hold information, e.g. "early access review" - these words are not part
    # of the user review
    @staticmethod
    def __remove_wasteful_words(text: str) -> str:
        new_text = text.replace("early access review", "")
        return new_text

    # Removing the whitespaces characters
    @staticmethod
    def __remove_blanks(value: str) -> str:
        return value.replace(" ", "")

    # After text preprocessing, some rows can be empty. This rows are removed by this method.
    @staticmethod
    def __drop_no_information(df: pd.DataFrame, column: str) -> pd.DataFrame:
        filtered_df = df[(df[column] != "") & (df[column] != None)]
        return filtered_df
    # ------------------------------------------------------------------------------------------------------------------
