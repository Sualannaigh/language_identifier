"""
metrics.py

This script generates metrics for evaluating the model for the
generated language model.

Command-line arguments:
  -c, --cross_validations: Specifies how many k-fold validations will
   be calculated
  -d, --data: Location to where the csv data will be read from.
   Default is "data/datasheet.csv".
  -n, --name: Name of the model. Default is "model".

Attributes:
- data:  The loaded data for training and evaluation.
- vectorizer: Vectorizer used in training the model
- categories: Encoder used in training the model.
- model: Trained language model.

Classes:
- Metrics: Class for interacting with the methods for generating metrics for
the language model.

Usage:
Run this script to generate metrics for a trained language model.

Example:
python3 metrics.py -c 5 -m gaussian_5000 -d big_dataset.csv
"""
import pickle
import sys
import argparse
from sklearn.model_selection import cross_val_score
import numpy as np
from model import Model

parser = argparse.ArgumentParser(
    description='This script generates metrics for evaluating the model for the'
    'generated language model.'
)
parser.add_argument(
    "-c",
    "--cross_validations",
    required=False,
    action="store",
    type=int,
    help="Specifies how many k-fold validations will be calculated",
)
parser.add_argument(
    "-d",
    "--data",
    required=False,
    action="store",
    default="data/datasheet.csv",
    type=str,
    help="Location to where the csv data will be read from",
)
parser.add_argument(
    "-n",
    "--name",
    required=False,
    action="store",
    default="model",
    help='Name of the model, default "model"',
)
args = parser.parse_args()


class Metrics:
    """
    A class for generating and evaluating metrics for the model.

    Attributes:
    - data: (pandas.DataFrame) The loaded data for training and evaluation.
    - vectorizer: Vectorizer used in training the model
    - categories: Encoder used in training the model.
    - model: Trained language model.

    Methods:
    - cross_validate()
        Generates cross-validation scores for the language prediction model
        using k-fold cross-validation.
    - get_top_features()
        Retrieves the top 10 words for predicting language.
    """

    def __init__(self) -> None:
        try:
            self.data = Model.load_data(args.data)
            with open("pickles/vectorizer.pickle", "rb") as vectorizer_file:
                self.vectorizer = pickle.load(vectorizer_file)
            with open("pickles/categories.pickle", "rb") as categories_file:
                self.categories = pickle.load(categories_file)
            with open(f"pickles/{args.name}.pickle", "rb") as model_file:
                self.model = pickle.load(model_file)
        except FileNotFoundError:
            print("To generate metrics you will need to train a model first.")
            sys.exit(1)
        self.cross_validate(args.cross_validations)
        self.get_top_features()

    def cross_validate(self, k_validations: int) -> None:
        """
        Generates cross-validation scores for the language prediction model
        using k-fold cross-validation.

        Args:
        - k_validations (int): How many validations will be evaluated.
        """
        print("Generating cross validation scores...")
        vectorized_data = self.vectorizer.fit_transform(
            self.data["text"].values.astype("U")
        )
        scores = cross_val_score(
            self.model, vectorized_data.toarray(), self.categories, cv=k_validations
        )
        print(f'Cross validation scores: {"% ".join(map(str, np.round(scores, 4)))}')

    def get_top_features(self) -> None:
        """
        Retrieves and prints the top 10 features based on the feature log
        probabilities of the model.
        """
        print("Top 10 words for predicting language")
        prob_sorted = self.model.feature_log_prob_[1, :].argsort()[::-1]
        top_features = np.take(
            self.vectorizer.get_feature_names_out(), prob_sorted[:10]
        )
        for position, feature in enumerate(top_features, start=1):
            print(f"{position}: {feature}")


if __name__ == "__main__":
    metrics = Metrics()
