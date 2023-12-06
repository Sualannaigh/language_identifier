"""
model.py

This script implements a naive bayesian classifier, with an interface for
training the model and predicting inputed text using the model.

Command-line arguments:
  -d, --data: Location to where the csv data will be read from.
   Default is "data/datasheet.csv".
  -t, --training: Enables training for the model.
  -n, --name: Name of the model. Default is "model".
  -f, --features: Number of features the vocabulary for the vectorizer will use
   Default is 5000 features.
  -p, --predict: Use input to predict text using the model.
  -v, --vectorizer: Sets the vectorizer method used for the program.
   Supported arguments are "tf" (TfidfVectorizer) and "cv" (CountVectorizer).
    Default is "cv".
  -c, --classifier: Choose which of the Naive Bayesian classifiers the model
   will use. Supported arguments are "b" (Bernoulli), "g" (Gaussian),
    and "m" (Multinomial). Default is "m".

Classes:
- Model: Class for interacting with the methods for generating model.

Usage:
Run this script to either train a new language identification model or predict
language for a given input text.

Example:
python3 model.py -t -f 5000 -c g -v tf -n gaussian_5000
"""

import os
import pickle
import time
import csv
import sys
import argparse
import pandas as pd
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

parser = argparse.ArgumentParser(description='This script implements a naive'
' bayesian classifier, with an interface for training the model and predicting'
' inputed text using the model.')
parser.add_argument(
    '-d',
    '--data',
    required=False,
    action='store',
    default='data/datasheet.csv',
    type=str,
    help='Location to where the csv data will be read from',
)
parser.add_argument(
    '-t',
    '--training',
    required=False,
    action='store_true',
    help='Enables training for the model',
)
parser.add_argument(
    '-n',
    '--name',
    required=False,
    action='store',
    default='model',
    help='Name of the model, default "model"',
)
parser.add_argument(
    '-f',
    '--features',
    required=False,
    action='store',
    default=5000,
    type=int,
    help='How many features the vocabulary for the vectorizer will use,'
    'default 5000 features',
)
parser.add_argument(
    '-p',
    '--predict',
    required=False,
    action='store',
    help='Use input to predict text using model',
)
parser.add_argument(
    '-v',
    '--vectorizer',
    required=False,
    action='store',
    default='cv',
    help='Sets the vectorizer method used for the program, supported arguments'
    'are "tf"(TfidfVectorizer) and "cv" (CountVectorizer). Default "cv".'
)
parser.add_argument(
    '-c',
    '--classifier',
    required=False,
    action='store',
    default='g',
    help='Choose which of the Naive Bayesian classifiers the model will use,'
    'supported arguments are "b" (Bernoullian),"g" (Gaussian) and "m"'
    '(Multinomial), default "m"'
)
ARGS = parser.parse_args()

# Mappings for the commandline arguments
VECTORIZER_DICT = {"cv": CountVectorizer, "tf": TfidfVectorizer}
CLASSIFIER_DICT = {"b": BernoulliNB, "g": GaussianNB, "m": MultinomialNB}


class Model:
    """
    Class containing a naive bayesian classifier for language identification.

    Attributes:
    - data_location (str): Location of the input data.
    - model (object): The trained machine learning model.
    """
    def __init__(self) -> None:
        """
        Initializes the Model instance and calls the prediction and/or
        training mode.
        """
        if (
            ARGS.vectorizer not in VECTORIZER_DICT
            or ARGS.classifier not in CLASSIFIER_DICT
        ):
            print(
                "Non supported parameter given for vectorizer or classifier"
                ", make sure you have inputed a correct argument"
            )
            sys.exit(1)
        self.data_location = ARGS.data
        self.model = None
        if ARGS.training:
            self.model = self.setup_training(self.data_location)
        if ARGS.predict:
            print(self.predict(ARGS.predict))

    @staticmethod
    def load_data(target_file: str) -> pd.DataFrame:
        """
        Loads data from .csv file and assures it exists

        ARGS:
        - target_file (str): Path to the csv file to load data from.
        
        Returns:
        - pd.DataFrame: Specified datafile.
        """
        print("Loading data...")
        if os.path.exists(target_file):
            return pd.read_csv(target_file, lineterminator="\n")
        print("The selected datafile could not be found, exiting program")
        sys.exit(1)

    def measure_classifier(self, input_model, test_set: list) -> Tuple[float, float]:
        """
        Measure the performance of the classifier.

        ARGS:
        - input_model (object): Trained naive bayes classifer.
        - test_set (list): List containing training data for text and language.

        Returns:
        - Tuple[float, float]: Accuracy and recall scores.
        """
        text, language = test_set
        language_prediction = input_model.predict(text)
        accuracy = accuracy_score(
            y_pred=language_prediction, y_true=language
        )
        accuracy = round(accuracy * 100, 4)
        recall = recall_score(
            y_pred=language_prediction, y_true=language, average="macro"
        )
        recall = round(recall * 100, 4)
        return accuracy, recall

    def train_model(self, train_data: np.ndarray, pred_data: np.ndarray) -> Tuple["model", float]:
        """
        Train the naive bayes classifier and times it.

        ARGS:
        - train_data (numpy.ndarray): Training data, composed of text paragraphs.
        - pred_data (numpy.ndarray): Prediction data, composed of languages.

        Returns:
        - Tuple[object, float]: Trained model and time to run model.
        """
        print("Starting training task...")
        nb_model = CLASSIFIER_DICT[ARGS.classifier]()
        start = time.time()
        nb_model.fit(train_data, pred_data)
        end = time.time()
        run_time = round(end - start, 2)
        return nb_model, run_time

    def setup_training(self, data_location: str) -> "model":
        """
        Setup training for the naive bayes classifier.

        ARGS:
        - data_location (str): Path to the input data.

        Returns:
        - object: Trained naive bayes classifier.
        """
        data = self.load_data(data_location)
        print("Vectorizing data and splitting datasets...")
        language = data["language"]
        encoder = LabelEncoder()
        # Encode the language labels (targets)
        categories = encoder.fit_transform(language)
        vectorizer = VECTORIZER_DICT[ARGS.vectorizer](max_features=ARGS.features)
        vectorized_data = vectorizer.fit_transform(data["text"].values.astype("U"))
        # Split up the dataset into training, and testing sets for text
        # and language
        text_train, text_test, language_train, language_test = train_test_split(
            vectorized_data.toarray(), categories, random_state=42
        )
        # Train model
        nb_model, run_time = self.train_model(text_train, language_train)
        precision, recall = self.measure_classifier(nb_model, [text_test, language_test])
        f_score = round(float(2 * (precision * recall) / (precision + recall)), 4)
        print(f'Fitting model took: {run_time} seconds with a precision of'
              f' {precision}%, recall of {recall}% and f1-score of {f_score}%'
        )
        # Pickling data to save for other uses within the program
        print("Pickling data...")
        with open("pickles/categories.pickle", "wb") as _categories:
            pickle.dump(categories, _categories)
        with open(f"pickles/{ARGS.name}.pickle", "wb") as _model:
            pickle.dump(nb_model, _model)
        with open("pickles/vectorizer.pickle", "wb") as _vectorizer:
            pickle.dump(vectorizer, _vectorizer)
        with open("pickles/encoder.pickle", "wb") as _encoder:
            pickle.dump(encoder, _encoder)
        self.write_log(run_time, precision, recall, f_score)
        return nb_model

    def predict(self, input_string: str) -> str:
        """
        Predict the language of an inputted string using a trained model.

        ARGS:
        - input_string (str): Text.

        Returns:
        - str: Predicted language.
        """
        with open(f"pickles/{ARGS.name}.pickle", "rb") as _nb_model:
            nb_model = pickle.load(_nb_model)
        with open("pickles/vectorizer.pickle", "rb") as _vectorizer:
            vectorizer = pickle.load(_vectorizer)
        with open("pickles/encoder.pickle", "rb") as _encoder:
            encoder = pickle.load(_encoder)
        # Vectorize the input and then transform back the language encoding
        vectorized_input = vectorizer.transform([input_string]).toarray()
        lang = encoder.inverse_transform(nb_model.predict(vectorized_input))
        return lang[0]

    def write_log(self, run_time: float, precision: float, recall: float, f_score: float) -> None:
        """
        Write training log to a csv file to compare results between
        different runs.

        ARGS:
        - run_time (float): Training runtime.
        - precision (float): Precision score.
        - recall (float): Recall score.
        - f_score (float): F1 score.
        """
        with open("pickles/data_ARGS.pickle", "rb") as _data_ARGS:
            data_ARGS = pickle.load(_data_ARGS)

        writing_ARGS = [
            ARGS.name,
            data_ARGS.data,
            precision,
            recall,
            f_score,
            run_time,
            ARGS.features,
            ARGS.vectorizer,
            ARGS.classifier,
        ]
        # Check if the log file exists
        log_path = "log.csv"
        if not os.path.exists(log_path):
            # If a log does not exist then create one with headers
            with open(log_path, "w", encoding="utf-8", newline='') as _log:
                csv_writer = csv.writer(_log)
                csv_writer.writerow([
                    "Name",
                    "Datapoints",
                    "Precision",
                    "Recall",
                    "F1",
                    "Time to run",
                    "Features",
                    "Vectorizer",
                    "Classifier",
                ])
                csv_writer.writerow(writing_ARGS)
        else:
            with open(log_path, "a", encoding="utf-8", newline='') as _log:
                csv_writer = csv.writer(_log)
                csv_writer.writerow(writing_ARGS)


if __name__ == "__main__":
    model_instance = Model()
