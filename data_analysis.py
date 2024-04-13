""" Author: Jule Godbersen (mailto:godbersj@tcd.ie)
    Content of file: Analysing a dataset and saving/returning the analysis results """

from datasets import load_dataset
import data_loading as data_loading
import pickle
from collections import Counter
import statistics


def get_sample_distributions(dataset):
    """Analyzes how the samples are distributed within the dataset

    Args:
        dataset (list): List of three dataset components: train,dev,test
    Returns:
        Result of analysing distribution of samples across splits
    """
    # analyze data split lengths
    len_entire_data = sum([len(split) for split in dataset])
    train = "\ttrain: \t"+str(round(len(dataset[0])/len_entire_data*100,1))+"%\t"+str(len(dataset[0]))+" instances"
    dev = "\tdev: \t"+str(round(len(dataset[1])/len_entire_data*100,1))+"%\t"+str(len(dataset[1]))+" instances"
    test = "\ttest: \t"+str(round(len(dataset[2])/len_entire_data*100,1))+"%\t"+str(len(dataset[2]))+" instances"

    return "Distribution of samples across data splits\n" + train + "\n" + dev + "\n" + test + "\n"

def get_label_distribution(datasplit,split:str):
    """Analyzes how samples are balanced across classes within a given datasplit
    Args:
        datasplit: data split to be investigated
        split (str): name of datasplit
    Returns:
        Result of analyzing label distribution within one split
    """
    neg = 0
    neut = 0
    pos = 0
    for sample in datasplit:
        if sample["label"] == 0:
            neg += 1
        elif sample["label"] == 1:
            neut += 1
        else:
            pos += 1
    all = neg + neut + pos
    return "\t"+split+": \t"+str(round(neg/(all)*100))+ "%\t["+str(neg)+"] negative instances\t\t" +str(round(neut/(all)*100))+ "%\t["+str(neut)+"] neutral instances\t\t" +str(round(pos/(all)*100))+ "%\t["+str(pos)+"] positive instances\t\t"

def get_label_distributions(dataset):
    """Analyzes distribution of labels across classes within entire dataset
    Args:
        dataset (list): List of three dataset components: train,dev,test
    Returns:
        Result of analyzing distribution of samples across classes in entire dataset
    """
    train = get_label_distribution(dataset[0],"train")
    dev = get_label_distribution(dataset[1],"dev")
    test = get_label_distribution(dataset[2],"test")
    return "Distribution of labels within data splits:\n" + train + "\n" + dev + "\n" + test + "\n"

def get_sources(dataset):
    train = "\ttrain: "+str(dict(Counter([sample["source"] for sample in dataset[0]])))
    dev = "\tdev: "+str(dict(Counter([sample["source"] for sample in dataset[1]])))
    test = "\ttest: "+str(dict(Counter([sample["source"] for sample in dataset[2]])))
    return "Distribution of data sources of the samples across the splits: \n" + train + "\n" + dev + "\n" + test + "\n"

# TODO evtl. include further analyses

def analyze_dataset(dataset,printing=False,path_for_report=None):
    """Helper function to automatically analyze a dataset; used for proof-checking if creation of data worked

    Args:
        dataset (list): List of three dataset components: train,dev,test
        printing (bool, optional): If true, results of analysis are printed (to terminal). Defaults to False.
        path_for_report (str, optional): If path to file is provided, a file is saved containing the results of the analysis. Defaults to None.
    """
    # call helper functions for analysis
    sample_distribution = get_sample_distributions(dataset)
    label_distribution = get_label_distributions(dataset)
    sources = get_sources(dataset)

    if printing:
        print(sample_distribution)
        print(label_distribution)
        print(sources)

    if path_for_report:
        analysis_report = open(path_for_report,"w")
        analysis_report.write(sample_distribution)
        analysis_report.write(label_distribution)
        analysis_report.write(sources)
        analysis_report.close()


# load data and analyze it
with open('/mount/studenten-temp1/users/godberja/GermanSentiment/data/ids.pkl', 'rb') as file:
    ids = pickle.load(file)
train_ids = ids["de_train_small"]
dev_ids = ids["de_dev_small"]
test_ids = ids["de_test"] 

# with open('/mount/studenten-temp1/users/godberja/GermanSentiment/data/translations.pkl', 'rb') as file:
#     translations = pickle.load(file)

dataset = load_dataset("Brand24/mms",cache_dir="/mount/studenten-temp1/users/godberja/HuggingfaceCache")
train_data = data_loading.load_own_dataset(dataset,train_ids,None)
dev_data = data_loading.load_own_dataset(dataset,dev_ids,None)
test_data = data_loading.load_own_dataset(dataset,test_ids,None)

# call the actual analysis
analyze_dataset([train_data,dev_data,test_data],True,"/mount/studenten-temp1/users/godberja/GermanSentiment/analysis/data.txt")