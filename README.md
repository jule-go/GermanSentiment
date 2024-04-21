# GermanSentiment
This project is done as part of the course Corpora in Speech and Language Processing in Trinity College Dublin.
It considers the task of sentiment analysis for German. I investigate whether using a Machine Translation system is beneficial for dealing with sentiment in a low-resource scenario.

For more details and insights on the results, please read through the submitted report.

In case of questions: Don't hesitate to email me!   [Jule Godbersen](mailto:godbersj@tcd.ie)

## Installation and Running
You need to install python (v3.8) and create a virtual environment (e.g. with help of python3.8 -m venv env_name). Clone this repository and install the requirements specified in the [``requirements.txt``-file](https://github.com/jule-go/GermanSentiment/blob/main/README.md) by using ``pip install -r requirements.txt``. For accessing the dataset on huggingface you need to have a huggingface account.

Before running any code snippets, make sure to activate the environment.

You might need to adapt the file paths in the individual files. The parts should be highlighted with a ``TODO``. 

Note: There might be additional ``TODO``s included. They highlight lines in which I was experimenting with different code snippets for the different scenarios.

[Note for me: To make sure the requirements.txt-file is up to date: Use ``pip freeze > GermanSentiment/requirements.txt`` when adding new packages.]

## Some information on folders/files within this repository
* [``analysis``](https://github.com/jule-go/GermanSentiment/blob/main/analysis) contains files proving that the data splits are balanced across labels and genres.
* [``data``](https://github.com/jule-go/GermanSentiment/blob/main/data) contains the datasplits I used and the translations I created. 
    * [``ids.pkl``](https://github.com/jule-go/GermanSentiment/blob/main/data/ids.pkl) a file listing the ids of the data splits.
    * [``translations.pkl``](https://github.com/jule-go/GermanSentiment/blob/main/data/translations.pkl) represents a saved dictionary with translations from German to English, and from English to German of the texts of the according data ids.
* [``evaluation``](https://github.com/jule-go/GermanSentiment/blob/main/evaluation) contains evaluation results, like confusion matrices of the trained models.
* [``predictions``](https://github.com/jule-go/GermanSentiment/blob/main/predictions) contains the model predictions on the test data. Note that the text and gold label are removed. But you can access this information by looking in the original dataset and searching for the according ID.
* [``training``](https://github.com/jule-go/GermanSentiment/blob/main/training) contains visualization of training the models for the different scenarios. (Note that the model checkpoints are not included as the files are too huge. But of course you can create them by running the training file with the hyperparams from the report. If still needed, please ask for the checkpoints via mail.)
* [``baseline.py``](https://github.com/jule-go/GermanSentiment/blob/main/baseline.py) Loads a germansentiment model and evaluates it on my test data.
* [``cleanup.ipynb``](https://github.com/jule-go/GermanSentiment/blob/main/cleanup.ipynb) Cleans the repo after evaluating models to make sure the data is not uploaded on the public repo.
* [``data.ipynb``](https://github.com/jule-go/GermanSentiment/blob/main/data.ipynb) deals with the preparation of the dataset(s). It includes balancing the datasets, creating train, dev and test splits, and translating the text instances.
* [``data_analysis.ipynb``](https://github.com/jule-go/GermanSentiment/blob/main/data_analysis.ipynb) provides functionality to analyze a dataset
* [``data_loading.py``](https://github.com/jule-go/GermanSentiment/blob/main/data_loading.py) makes sure we can load the dataset contents into the correct formats needed as input for the models.
* [``evaluation.py``](https://github.com/jule-go/GermanSentiment/blob/main/evaluation.py) lets one evaluate the performance of a model on test data
* [``model_definition.py``](https://github.com/jule-go/GermanSentiment/blob/main/model_definition.py) defines the architecture of the model used within this project
* [``requirements.txt``](https://github.com/jule-go/GermanSentiment/blob/main/requirements.txt) contains requirements for python environment that are needed to run the code within this repository.
* [``training.py``](https://github.com/jule-go/GermanSentiment/blob/main/training.py) lets one train a model on training data
* [``training_logging.md``](https://github.com/jule-go/GermanSentiment/blob/main/training_logging.md) provides important parts of the overview of the hyperparameter tuning
