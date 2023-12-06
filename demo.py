"""
This is a very short script to showcase the model being used "in-action" by
loading in a saved model (with the name model.pickle located in the pickles
directory) and predicting the language of all the files located in the
directory ~/example_data~.

To use this script you will first have to follow the previous steps of training
a model, making sure there is a file named model.pickle within the
"/pickles" directory.
"""
import os
from model import Model


model = Model()

for text in os.listdir("example_data/"):
    file_path = os.path.join("example_data", text)
    with open(file_path, "r+", encoding="utf-8") as file:
        content = file.read()
        print(f"'{text}' predicted to be: {model.predict(content)}")
