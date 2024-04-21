""" Author: Jule Godbersen (mailto:godbersj@tcd.ie)
    Content of file: Evaluation of baseline model; similar to evaluation.py, only slight changes """

import os
import torch
from torch import nn
import data_loading as data_loading
import pickle
import numpy as np
from datasets import load_dataset
from sklearn import metrics
import statistics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import json
from germansentiment import SentimentModel


# -----------------------------------------------------------------------------------------
# [in this section one should change the values!]

# specify location of train/validation data
with open('/mount/studenten-temp1/users/godberja/GermanSentiment/data/ids.pkl', 'rb') as file:
    ids = pickle.load(file)
dev_ids = ids["de_dev_small"]
test_ids = ids["de_test"]

# load translations
with open('/mount/studenten-temp1/users/godberja/GermanSentiment/data/translations.pkl', 'rb') as file:
    translations = pickle.load(file)

# specify whether we need to look up translations
translation_dev = None
translation_test = None
test_identifier = "german_test" # e.g. indicate for myself whether test data got translated


# specify where to save predictions to
prediction_path = "/mount/studenten-temp1/users/godberja/GermanSentiment/evaluation/predictions_" +test_identifier+"_baseline.pkl" 

# load model
model = SentimentModel()
print("Model is loaded and ready to be evaluated")

# set some hyperparams 
batch_size = 32  # maybe try 32, 16, 64
loss_function = nn.CrossEntropyLoss() # nn.L1Loss() # nn.MSELoss() # nn.CrossEntropyLoss()


# -----------------------------------------------------------------------------------------
# [don't change things in this section]

device = torch.device("cpu")

os.environ["HF_DATASETS_CACHE"] = "/mount/studenten-temp1/users/godberja/HuggingfaceCache" # TODO needed?
os.environ["TRANSFORMERS_CACHE"] = "/mount/studenten-temp1/users/godberja/Cache" # TODO needed?

# load the data
dataset = load_dataset("Brand24/mms",cache_dir="/mount/studenten-temp1/users/godberja/HuggingfaceCache")
dev_data = data_loading.load_own_dataset(dataset,dev_ids,translation_dev)
dev_dataloader = data_loading.load_dataloader(dev_data,batch_size)
print("Development data is loaded")
test_data = data_loading.load_own_dataset(dataset,test_ids,translation_test)
test_dataloader = data_loading.load_dataloader(test_data,batch_size)
print("Test data is loaded")


def test(model, test_data, loss_function, device):
    """Tests a model with the test data

    Args: (note: not sure with all the types here ^^)
        model (nn.Module): Model that was trained within this epoch and now should be tested
        test_data (torch.utils.Data.DataLoader): Data used for testing the model performance
        loss_function (torch.nn): Function that should be used for calculating the loss
        device (torch.device): GPU device on which the model should run

    Returns:
        (float,float,float): Average accuracy of this epoch, Average loss of this epoch, F-score
    """

    losses,corrects,total = 0,0,0
    pred_labels_all = []
    gold_labels_all = []

    for batch in test_data:
        texts = batch["texts"]

        labels = torch.stack(tuple([torch.nn.functional.one_hot(torch.tensor(int(label_test)),3) for label_test in batch["labels"]])) # stacked tensor of labels (one-hot encoded)
        labels = labels.type(torch.float32)
        labels = labels.to(device)
        gold_labels = torch.argmax(labels,dim=1)
        gold_list = gold_labels.tolist()
        for label in gold_list:
            gold_labels_all.append(label)

        # here we call the "baseline" model
        _, probabilities = model.predict_sentiment(texts,output_probabilities=True)
        output = []
        for pred in probabilities:
            output_temp = [0,0,0]
            for (label,prob) in pred:
                # print(label,prob)
                if label == "negative":
                    output_temp[0] = prob
                elif label == "neutral":
                    output_temp[1] = prob
                else: # label == "positive"
                    output_temp[2] = prob
            output += [output_temp]
        output = torch.tensor(output)

        loss = loss_function(output, labels)

        losses += loss.item()
        pred_probs = torch.softmax(output,dim=1) # apply softmax in case a model does not have softmax as final layer
        pred_labels = torch.argmax(pred_probs,dim=1)
        pred_list = pred_labels.tolist()
        for label in pred_list:
            pred_labels_all.append(label)

        correct = torch.sum(torch.eq(pred_labels,gold_labels))
        corrects += correct.item()
        total += len(texts)

    avg_acc = corrects / total 
    f_score = metrics.f1_score(gold_labels_all, pred_labels_all,average="weighted")
    prec, rec, fbeta, support = metrics.precision_recall_fscore_support(gold_labels_all, pred_labels_all)

    avg_loss = losses / len(test_data) # consider amount of different batches

    print(f"Loss: {avg_loss}, Accuracy: {avg_acc}, F1-Score: {f_score}, Precision: {prec}, Recall: {rec}")

    return avg_acc, avg_loss, f_score


def make_predictions(test_dataset,model,path_to_save_prediction_to):
    """Uses model to make prediction on test data and saves them to file

    Args:
        test_dataset (torch.utils.Data.DataLoader): Data for testing
        model (nn.Module): Model that is used for doing the predictions
        path_to_save_prediction_to (str): Path where the predictions are being saved to
    """
    predictions = []
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for datapoint in test_dataset:
            prediction = dict()

            id = datapoint["id"]
            text = datapoint["text"] 
            gold_label = datapoint["label"] 
            gold_tensor = torch.stack(tuple([torch.nn.functional.one_hot(torch.tensor(int(gold_label)),num_classes=3)]))
            gold_tensor = gold_tensor.type(torch.float32)
            gold_tensor = gold_tensor.to(device)
            prediction["id"] = id
            prediction["text"] = text
            prediction["gold"] = str(gold_label)

            # use "baseline" model here
            _, probabilities = model.predict_sentiment([text],output_probabilities=True)
            model_pred = []
            for pred in probabilities:
                output_temp = [0,0,0]
                for (label,prob) in pred:
                    # print(label,prob)
                    if label == "negative":
                        output_temp[0] = prob
                    elif label == "neutral":
                        output_temp[1] = prob
                    else: # label == "positive"
                        output_temp[2] = prob
                model_pred += [output_temp]
            model_pred = torch.tensor(model_pred)

            model_pred_prob = torch.softmax(model_pred,dim=1)
            model_pred_label = torch.argmax(model_pred_prob,dim=1)
            prediction["pred"] = str(model_pred_label.item())
            prediction["prob"] = [round(val,3) for val in model_pred_prob[0].tolist()]

            predictions.append(prediction)
        
    with open(path_to_save_prediction_to,"wb") as outfile:
        pickle.dump(predictions,outfile)   

def validateAndTest(model,dev_dataloader,test_dataloader,loss_function,test_data,prediction_path,device):
    """As I won't save the model to the hub, here I want to directly evaluate it by applying it on dev and test data, and saving the predictions to file

    Args:
        model (nn.Module): Model that is evaluated by checking for its performance on validation and test data, predictions on test data are getting saved
        dev_dataloader (torch.utils.Data.DataLoader): Data used for validating the training process
        test_dataloader (torch.utils.Data.DataLoader): Data used for evaluating the model performance
        loss_function (torch.nn): Function that should be used for calculating the loss
        test_data (torch.utils.Data.DataLoader): Data used for testing the model performance
        prediction_path (str): Path where the predictions are being saved to
        device (torch.device): GPU device on which the model should run
    """
    # apply model on dev data
    print("\nvalidation on dev:")
    test(model, dev_dataloader, loss_function, device)

    # apply model / make predictions on test data
    print("\nevaluation on test:")
    test(model, test_dataloader, loss_function, device)
    make_predictions(test_data,model,prediction_path)


def analyze_quantitatively(predictions, dataset, printing=False, path_to_save_evaluation_to=None):
    """Analyzes the predictions of a model quantitatively, e.g. looking at its accuracy, generating a confusion matrix

    Args:
        predictions (list): List of items, where each item has an id, text, predicted label, gold label, probability distribution over prediction
        dataset: Dataset from Huggingface with all the data sampels
        printing (bool, optional): If true, it prints the results of this analysis to terminal. Defaults to False.
        path_to_save_evaluation_to (str, optional): If path is provided, the results of this analysis are saved to this file. Defaults to None.
    """

    # investigate general performance of model
    preds = [datapoint["pred"] for datapoint in predictions]
    golds = [datapoint["gold"] for datapoint in predictions]
    accuracy = "Accuracy: "+str(metrics.accuracy_score(golds, preds))+"\n"
    f_score = "F1-score: "+str(metrics.f1_score(golds, preds, average="weighted"))+"\n"

    # investigate confidence of model
    label_probs = [max(datapoint["prob"]) for datapoint in predictions]
    avg_prob = round(sum(label_probs) / len(label_probs),3)
    min_prob = min(label_probs)
    max_prob = max(label_probs)
    median_prob = round(statistics.median(label_probs),3)
    std_prob = round(statistics.stdev(label_probs),3)
    confidence = "Confidence of model: "+str(avg_prob)+"[avg]\t"+str(std_prob)+"[std dev]\t"+str(min_prob)+"[min]\t"+str(max_prob)+"[max]\t"+str(median_prob)+"[median]\n"

    # generate a confusion matrix 
    conf_matrix = np.array2string(metrics.confusion_matrix(golds,preds,labels=["0","1","2"]))
    confusion = "Confusion matrix: \n"+str(conf_matrix)+"\n"
    ConfusionMatrixDisplay.from_predictions(golds,preds)
    plt.title(str("Confusion Matrix"))
    if path_to_save_evaluation_to:
        plt.savefig(str(path_to_save_evaluation_to+"_confusionMatrix.jpg")) 
    plt.close()

    # look at difference across "genre"
    ids = [datapoint["id"] for datapoint in predictions]
    sources = [dataset["train"][data_id]["original_dataset"] for data_id in ids]
    twitter_preds = []
    twitter_golds = []
    review_preds = []
    review_golds = []
    for i, src in enumerate(sources):
        if "amazon" in src:
            review_preds += [predictions[i]["pred"]]
            review_golds += [predictions[i]["gold"]]
        else: # "twitter" in src:
            twitter_preds += [predictions[i]["pred"]]
            twitter_golds += [predictions[i]["gold"]]
    twitter_accuracy = "Accuracy on twitter data only: "+str(metrics.accuracy_score(twitter_golds, twitter_preds))+"\n"
    review_accuracy = "Accuracy on review data only: "+str(metrics.accuracy_score(review_golds, review_preds))  +"\n"
    conf_matrix_twitter = np.array2string(metrics.confusion_matrix(twitter_golds,twitter_preds,labels=["0","1","2"]))
    confusion_twitter = "Confusion matrix of twitter data: \n"+str(conf_matrix_twitter)+"\n"
    plt.clf()
    ConfusionMatrixDisplay.from_predictions(twitter_golds,twitter_preds)
    plt.title(str("Confusion Matrix Twitter data"))
    if path_to_save_evaluation_to:
        plt.savefig(str(path_to_save_evaluation_to+"_confusionMatrix_twitter.jpg")) 
    plt.close()
    conf_matrix_review = np.array2string(metrics.confusion_matrix(review_golds,review_preds,labels=["0","1","2"]))
    confusion_review = "Confusion matrix of review data: \n"+str(conf_matrix_review)+"\n"
    plt.clf()
    ConfusionMatrixDisplay.from_predictions(review_golds,review_preds)
    plt.title(str("Confusion Matrix Review data"))
    if path_to_save_evaluation_to:
        plt.savefig(str(path_to_save_evaluation_to+"_confusionMatrix_review.jpg")) 
    plt.close()

    # analyze the model performance with the built-in classification_report function
    report = "Built-in analysis: \n"+str(metrics.classification_report(y_true=golds,y_pred=preds))+"\n"

    if printing:
        print(accuracy)
        print(f_score)
        print(confidence)
        print(confusion)
        print(report)
        print("\nLet's look at differences between 'genres':")
        print(twitter_accuracy)
        print(review_accuracy)
        print(confusion_twitter)
        print(confusion_review)
    
    if path_to_save_evaluation_to:
        analysis_report = open(str(path_to_save_evaluation_to+".txt"),"w")
        analysis_report.write(accuracy)
        analysis_report.write(f_score)
        analysis_report.write(confidence)
        analysis_report.write(confusion)
        analysis_report.write(report)
        analysis_report.write("\nLet's look at differences between 'genres':\n")
        analysis_report.write(twitter_accuracy)
        analysis_report.write(review_accuracy)
        analysis_report.write(confusion_twitter)
        analysis_report.write(confusion_review)
        analysis_report.close()


def order_samples_for_qualitative_analysis(predictions, translations, path_to_file):
    """Displays predictions in a more human-readable way

    Args:
        predictions (list): List of items, where each item has an id, text, predicted label, gold label, probability distribution over prediction
        translations (dict): key = text in source language, value = text in translated target language
        path_to_file (str): Reordering is saved to the file specified via this path
    """
    # order
    situations = [[[],[],[]],[[],[],[]],[[],[],[]]] # outer index represents gold, inner index prediction
    for pred in predictions:
        if pred["text"] in translations:
            pred["translated"] = translations[pred["text"]]
        outer_ind = int(pred["gold"])
        inner_ind = int(pred["pred"])
        situations[outer_ind][inner_ind] += [pred]
    # save to file
    situations_dict = {"gold: neg":{"pred: neg":situations[0][0],
                                    "pred: neut":situations[0][1],
                                    "pred: pos":situations[0][2]},
                       "gold: neut":{"pred: neg":situations[1][0],
                                    "pred: neut":situations[1][1],
                                    "pred: pos":situations[1][2]},
                        "gold: pos":{"pred: neg":situations[2][0],
                                    "pred: neut":situations[2][1],
                                    "pred: pos":situations[2][2]}}
    with open(path_to_file,"w") as file:
        json.dump(situations_dict,file,indent=3)


# -----------------------------------------------------------------

# evaluate model performance and save predictions to file
validateAndTest(model,dev_dataloader,test_dataloader,loss_function,test_data,prediction_path,device)

# analyze predictions
with open(prediction_path, 'rb') as file:
    predictions = pickle.load(file)
analyze_quantitatively(predictions,dataset,True,"/mount/studenten-temp1/users/godberja/GermanSentiment/evaluation/baseline_model")
order_samples_for_qualitative_analysis(predictions,translations,"/mount/studenten-temp1/users/godberja/GermanSentiment/evaluation/baseline_model_qualitative.json")