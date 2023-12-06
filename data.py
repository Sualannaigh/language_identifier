"""
data.py

This script collects and processes corpora files located in the directory
"data" for further use in the script model.py. It creates a csv files located
in the folder "data". The script works by extracting data in <p> tags found in
the documents, so it will work with most xml files of similar structures.

Command-line arguments:
  -s, --size: Lines of data to include in the model (for both languages).
   Default is 30 000 lines.
  -d, --data: Sets the name for the dataset. Default is "datasheet".

Classes:
- Gatherer: Represents a data gatherer for language identification.

Usage:
Run this script to collect and process text data from xml files, creating
a dataset for training a naive bayesian classifier.

Example:
python3 data.py -s 50000 -d big_dataset

This will create a dataset named "big_dataset.csv" with 50000 lines of data.
"""
import csv
import os
import re
import argparse
import pickle
import xml.etree.ElementTree as ET
import pandas as pd

AVOID_PATHS = {".git", "tmx", ".gitignore"}
CSV_HEADER = ["text", "language"]

parser = argparse.ArgumentParser(description='This script collects and processes'
' corpora files located in the directory "data" for further use in the script'
'model.py. It creates a csv files located in the folder "data". The script'
' works by extracting data in <p> tags found in the documents, so it will work'
' with most xml files of similar structures.')
parser.add_argument(
    '-s',
    '--size',
    required=False,
    action='store',
    type=int,
    default=30000,
    help='Lines of data to include in the model (for both language),'
    ' default 30000',
)
parser.add_argument(
    '-d',
    '--data',
    required=False,
    action='store',
    default='datasheet',
    type=str,
    help='Sets the name for the dataset, default "datasheet"',
)

args = parser.parse_args()


class Gatherer:
    """
    Class containing methods for extracting texts from corpora and saving them
    to a csv format.

    Attributes:
    - files (list): Location of the input data.
    - model (object): The trained machine learning model.
    - languages (list): List of languages (read from /data folder)
    - data_location (str) : Where the extracted data will be saved to
    """
    def __init__(self) -> None:
        """
        Initializes the Gatherer instance, calls methods for collecting data
        and saves the data as a csv file. It also pickles the commandline
        arguments so they can be used in logging purposes in model.py
        """
        self.files = []  # list of tuples which contains the data
        self.languages = [
            d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))
        ]
        self.language_weights = {}
        if (
            input(
                "Set ratio of languages? (if you do not the program will"
                " create an equal split between all languages)\ny/N: "
            )
            == "y"
        ):
            print(
                "Set a value between 0.0 - 1.0, e.g a value of 0.1 means that"
                " 10% of the final training-evaluation dataset will be of "
                "chosen language. The chosen values should add up to 1.0 (100%)"
            )
            for lang in self.languages:
                self.language_weights[lang] = float(input(f"Weight of {lang}: "))
        else:
            # Assign equal weights to all languages
            for lang in self.languages:
                self.language_weights[lang] = 1 / len(self.languages)
        self.data_location = f"data/{args.data}.csv"
        self.collect_text(self.languages)
        adjusted_df = self.adjust_dataset(self.data_location)
        adjusted_df.to_csv(f"data/{args.data}.csv")
        # Pickles commandline arguments so they can be used for logging
        if not os.path.exists("pickles"):
            os.makedirs("pickles")
        with open("pickles/data_args.pickle", "wb") as _args:
            pickle.dump(args, _args)
        print("Finished, you can now train with 'model.py'")

    def adjust_dataset(self, data_location: str) -> pd.DataFrame:
        """
        Reads from a csv file containing all data taken from the corpora and
        extracts data for the final data set, adjusting the weight of each
        language and shuffling the dataset.

        Args:
        - data_location (str): Path to the input CSV file.

        Returns:
        - pd.DataFrame: Adjusted dataset.
        """
        print("Tweaking data...")
        data_frame = pd.read_csv(data_location)
        dataframes = []
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
        final_size = args.size
        # Calculate how much data will be sampled from each language, and
        # check if there is enough data, else reduce datasize.
        decrease_percentage = 1
        for language in self.languages:
            corpus_share = int(final_size * self.language_weights[language])
            language_size = len(data_frame[data_frame.language == language])
            if language_size < corpus_share:
                if language_size / corpus_share < decrease_percentage:
                    decrease_percentage = language_size / corpus_share
                    print(
                        f'Language "{language}" does not have enough data'
                        f' ({language_size}/{corpus_share} requested paragraphs).'
                        ' Reducing size of final dataset to '
                        f'{round(decrease_percentage * 100, 2)}% of'
                        ' requested size)'
                    )
                    final_size *= decrease_percentage
        for language in self.languages:
            language_size = int(final_size * self.language_weights[language])
            print(f"Adding {language_size} paragraphs for {language}")
            dataframes.append(data_frame[data_frame.language == language].head(language_size))
        new_data_frame = pd.DataFrame(columns=CSV_HEADER)
        for frame in dataframes:
            new_data_frame = pd.concat([new_data_frame, frame], ignore_index=True)
        # Shuffle dataset again and return it
        return new_data_frame.sample(frac=1).reset_index(drop=True)

    def collect_text(self, languages: list) -> None:
        """
        Collects text data from xml corpora files for the specified languages
        and outputs it to a csv file.

        Args:
        - languages (list): List of languages to include in the dataset.
        """
        with open(self.data_location, "w", encoding="utf-8") as out_file:
            csv_writer = csv.writer(out_file)
            csv_writer.writerow(CSV_HEADER)
            print("Finding corpus files...")
            for language in languages:
                target_path = os.path.join(os.curdir, "data", language)
                assert target_path
                self.find_files(target_path, language)
            print("Building csv from gathered corpora...")
            for current_file in self.files:
                for parsed_text in self.parse_xml(current_file[0]):
                    if parsed_text.text:
                        processed_text = self.process_text(parsed_text.text)
                        for result in processed_text:
                            if result != "":
                                csv_writer.writerow([result, current_file[1]])

    def find_files(self, input_path: str, current_lang: str) -> None:
        """
        Finds xml files recursively in a given directory, discards directories
        listed in variable _avoidPaths.

        Args:
        - input_path (str): Directory path to search for files and directories.
        - current_lang (str): Current language being processed.
        """
        for current_path in os.listdir(input_path):
            if current_path not in AVOID_PATHS:
                current_path = os.path.join(input_path, current_path)
                if os.path.isfile(current_path) and current_path.endswith(".xml"):
                    self.files.append((current_path, current_lang))
                if os.path.isdir(current_path):
                    self.find_files(current_path, current_lang)

    def parse_xml(self, input_file: str) -> list:
        """
        Parses xml files and extracts the content of <p> tags.

        Args:
        - input_file (str): Directory path to te xml filed being extracted.

        Returns:
        - list: List of strings extracted from <p> tags in the document.
        """
        current_tree = ET.parse(input_file)
        root = current_tree.getroot()
        p_tags = root.findall(".//p")
        return p_tags

    def process_text(self, input_text: str) -> list:
        """
        Processes input text by removing unwanted characters and whitespaces.

        Args:
        - input_text (str): Input text.

        Returns:
        - list: Processed text.
        """
        result = re.sub(r"[^a-zA-Z\säåöÖÄÅ]", "", input_text)
        result = re.sub(r"\s+", " ", result).strip()
        return [result]


if __name__ == "__main__":
    gatherer = Gatherer()
