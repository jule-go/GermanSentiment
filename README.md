# GermanSentiment
This project is done as part of the course Corpora in Speech and Language Processing in Trinity College Dublin.
It considers the task of sentiment analysis for German. I investigate whether using a Machine Translation system is beneficial for dealing with sentiment in a low-resource scenario.

In case of questions: Don't hesitate to email me!   [Jule Godbersen](mailto:godbersj@tcd.ie)

## Installation and Running
You need to install python (v3.8) and create a virtual environment (e.g. with help of python3.8 -m venv env_name). Clone this repository and install the requirements specified in the [``requirements.txt``-file](https://github.com/jule-go/GermanSentiment/blob/main/README.md) by using ``pip install -r requirements.txt``. For accessing the dataset on huggingface you need to have a huggingface account.

Before running any code snippets, make sure to activate the environment.

[Note for me: To make sure the requirements.txt-file is up to date: Use ``pip freeze > GermanSentiment/requirements.txt`` when adding new packages.]


## Submissions
Proposal: [``GODBERSJ-23377840_proposal.pdf``](https://github.com/jule-go/GermanSentiment/blob/main/GODBERSJ-23377840_proposal.pdf)

Final Report: [``GODBERSJ-23377840.pdf``](https://github.com/jule-go/GermanSentiment/blob/main/GODBERSJ-23377840.pdf)

## Some information on files within this repository
* [``data.ipynb``](https://github.com/jule-go/GermanSentiment/blob/main/data.ipynb) deals with the preparation of the dataset(s). It includes balancing the datasets, creating train, dev and test splits, and translating the text instances.
* [``data_analysis.ipynb``](https://github.com/jule-go/GermanSentiment/blob/main/data_analysis.ipynb) provides functionality to analyze a dataset
* [``data_loading.py``](https://github.com/jule-go/GermanSentiment/blob/main/data_loading.py) makes sure we can load the dataset contents into the correct formats needed as input for the models.
* [``evaluation.py``](https://github.com/jule-go/GermanSentiment/blob/main/evaluation.py) lets one evaluate the performance of a model on test data
* [``model_definition.py``](https://github.com/jule-go/GermanSentiment/blob/main/model_definition.py) defines the architecture of the model used within this project
* [``requirements.txt``](https://github.com/jule-go/GermanSentiment/blob/main/requirements.txt) contains requirements for python environment that are needed to run the code within this repository.
* [``training_logging.md``](https://github.com/jule-go/GermanSentiment/blob/main/training_logging.md) provides an overview of the hyperparameter tuning
* [``training.py``](https://github.com/jule-go/GermanSentiment/blob/main/training.py) lets one train a model on training data
