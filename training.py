""" Author: Jule Godbersen (mailto:godbersj@tcd.ie)
    Content of file: Script for training the model(s)"""

import os
import torch
from torch import nn
import data_loading as data_loading
import pickle
from model_definition import Classifier
import numpy as np
from datasets import load_dataset
from sklearn import metrics
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------------------
# [in this section one should change the values!]

# specify on which GPU you want to run the model
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# specify location of train/validation data
with open('/mount/studenten-temp1/users/godberja/GermanSentiment/data/ids.pkl', 'rb') as file:
    ids = pickle.load(file)
    
train_ids = ids["de_train_small"]
print("Use ",len(train_ids)," datasamples for training")
train_identifier = "german_train_small"
dev_ids = ids["de_dev_small"]
print("Use ",len(dev_ids)," datasamples for evaluation")
dev_identifier = "german_dev_small"

# specify whether we need to look up translations
translation_train = None
translation_dev = None

# specifiy whether to optimize on accuracy (True) or on F1-Score
acc_bool = True #False

# specify where to save the model later
training_id = 0 # only used for us for logging the different hyperparameter effects
saving_path = "/mount/studenten-temp1/users/godberja/GermanSentiment/models/model_" + str(training_id) + ".pt"

# specify where you want to keep track of hyperparameter tuning
logging_file = "/mount/studenten-temp1/users/godberja/GermanSentiment/training_logging.md"

# adapter configuration that should be loaded
adapter_config = "lora"

# specify "size" of classification network
classification_layer_size = 100 # 75, 15, 100, 65, 120

# decide for activation function
activation = nn.ReLU()

# load model
model = Classifier(adapter_config=adapter_config,layer_size=classification_layer_size,activation=activation,device=device) 
model = model.to(device) 
print("Model is loaded")

# set some hyperparams for training 
num_epochs = 10 # 40, 3, 20, 100
batch_size = 8  # maybe try 32, 16, 64
learning_rate = 1e-3 # maybe try 1e-4 # 5e-4
loss_function = nn.MSELoss() # nn.L1Loss() # nn.MSELoss() # nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate) # AdamW, SGD, Adam, RMSProp

# specify random seed
seed = 99  #24, 99

# -----------------------------------------------------------------------------------------
# [don't change things in this section]

os.environ["HF_DATASETS_CACHE"] = "/mount/studenten-temp1/users/godberja/HuggingfaceCache" # TODO needed?
os.environ["TRANSFORMERS_CACHE"] = "/mount/studenten-temp1/users/godberja/Cache" # TODO needed?

print("Number of total params: ", sum(p.numel() for p in model.parameters()))
print("Number of trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Training is running on device: ",device)

# set the seeds 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# load the data
dataset = load_dataset("Brand24/mms",cache_dir="/mount/studenten-temp1/users/godberja/HuggingfaceCache")
train_data = data_loading.load_own_dataset(dataset,train_ids,translation_train)
train_dataloader = data_loading.load_dataloader(train_data,batch_size)
print("Training data is loaded")
dev_data = data_loading.load_own_dataset(dataset,dev_ids,translation_dev)
dev_dataloader = data_loading.load_dataloader(dev_data,batch_size)
print("Development data is loaded")



def train_epoch(model,train_data,optimizer,loss_function,device):
    """Trains a model for one epoch with the training data

    Args: (note: not sure with all the types here ^^)
        model (nn.Module): Model that should be trained
        train_data (torch.utils.Data.DataLoader): Data used for training
        optimizer (torch.optim): Optimizer that should be used for optimizing the model
        loss_function (torch.nn): Function that should be used for calculating the loss
        device (torch.device): GPU device on which the model should run

    Returns:
        (float,float): Average accuracy of this epoch, Average loss of this epoch
    """
    model.train()
    losses,corrects,total = 0,0,0
    pred_labels_all = []
    gold_labels_all = []
    for batch in train_data:
        optimizer.zero_grad()
        svo_triplets = batch["texts"]

        labels = torch.stack(tuple([torch.nn.functional.one_hot(torch.tensor(int(label_val)),3) for label_val in batch["labels"]])) # stacked tensor of labels (one-hot encoded)
        labels = labels.type(torch.float32)
        labels = labels.to(device)
        gold_labels = torch.argmax(labels,dim=1)
        gold_list = gold_labels.tolist()
        for label in gold_list:
            gold_labels_all.append(label)

        loss,output = model(svo_triplets,labels,loss_function)

        losses += loss.item()
        loss.backward()
        pred_probs = torch.softmax(output,dim=1) # apply softmax in case a model does not have softmax as final layer
        pred_labels = torch.argmax(pred_probs,dim=1)
        pred_list = pred_labels.tolist()
        for label in pred_list:
            pred_labels_all.append(label)

        correct = torch.sum(torch.eq(pred_labels,gold_labels))
        corrects += correct.item()
        total += len(svo_triplets)

        optimizer.step()

    avg_acc = corrects / total
    f_score = metrics.f1_score(gold_labels_all, pred_labels_all,average="weighted")
        
    avg_loss = losses / len(train_data) # consider amount of different batches

    return avg_acc, avg_loss, f_score

def validate_epoch(model,val_data,loss_function,device):
    """Validates a model for one epoch with the validation data

    Args: (note: not sure with all the types here ^^)
        model (nn.Module): Model that should be trained
        val_data (torch.utils.Data.DataLoader): Data used for validating the training process
        loss_function (torch.nn): Function that should be used for calculating the loss
        device (torch.device): GPU device on which the model should run

    Returns:
        (float,float): Average accuracy of this epoch, Average loss of this epoch
    """
    model.eval()
    losses,corrects,total = 0,0,0
    pred_labels_all = []
    gold_labels_all = []
    for batch in val_data:
        svo_triplets = batch["texts"]

        labels = torch.stack(tuple([torch.nn.functional.one_hot(torch.tensor(int(label_val)),3) for label_val in batch["labels"]])) # stacked tensor of labels (one-hot encoded)
        labels = labels.type(torch.float32)
        labels = labels.to(device)
        gold_labels = torch.argmax(labels,dim=1)
        gold_list = gold_labels.tolist()
        for label in gold_list:
            gold_labels_all.append(label)

        loss,output = model(svo_triplets,labels,loss_function)

        losses += loss.item()
        pred_probs = torch.softmax(output,dim=1) # apply softmax in case a model does not have softmax as final layer
        pred_labels = torch.argmax(pred_probs,dim=1)
        pred_list = pred_labels.tolist()
        for label in pred_list:
            pred_labels_all.append(label)

        correct = torch.sum(torch.eq(pred_labels,gold_labels))
        corrects += correct.item()
        total += len(svo_triplets)

    avg_acc = corrects / total 
    f_score = avg_acc = metrics.f1_score(gold_labels_all, pred_labels_all,average="weighted")

    avg_loss = losses / len(train_data) # consider amount of different batches

    return avg_acc, avg_loss, f_score


def plot_training(train_metric_values,train_losses,val_metric_values,val_losses,path_to_save_plot_to, metric):
    """Plot the accuracy and loss values collected during the training process

    Args:
        train_metric_values (list): Accuracies or F1-Score on training split
        train_losses (list): Losses on training split
        val_metric_values (list): Accuracies or F1-Score on validation split
        val_losses (list): Losses on validation split
        path_to_save_plot_to (str): Specifies where the plot should be saved e.g. cute-cats/Models/RobertaModel.png
        metric (str): Must be "Accuracy" or "F1-Score"
    """
    epochs = np.arange(1,len(train_metric_values)+1)
    
    train_metric_values = np.array(train_metric_values)
    train_losses = np.array(train_losses)
    val_metric_values = np.array(val_metric_values)
    val_losses = np.array(val_losses)

    # plot lines with accuracies / F1-Score -> consider left y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(metric)
    ax1.plot(epochs, train_metric_values, label=f"Training {metric}", color="blue")
    ax1.plot(epochs, val_metric_values, label=f"Validation {metric}", linestyle='dashed', color="red")
    ax1.scatter(epochs, train_metric_values, color="blue")
    ax1.scatter(epochs, val_metric_values, color="red")
    ax1.tick_params(axis='y')
    ax1.legend(loc=0)

    # plot lines with losses -> consider right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    ax2.plot(epochs, train_losses, label='Training Loss', color="lightblue")
    ax2.plot(epochs, val_losses, label='Validation Loss', linestyle='dashed', color="lightcoral")
    ax2.scatter(epochs, train_losses, color="lightblue")
    ax2.scatter(epochs, val_losses, color="lightcoral")
    ax2.tick_params(axis='y')
    ax2.legend(loc=1)

    fig.suptitle(f"{metric} and Loss Over Epochs") # add overall title

    plt.savefig(path_to_save_plot_to, bbox_inches='tight') # save the plot to file


def train(model,train_data,val_data,num_epochs,optimizer,loss_function,device,path_to_save_model_to):
    """Trains a model according to the specific hyperparameters with the training data, by iteratively doing epochs of training.
       For comparison it also validates on the specified validation data.

    Args:
        model (nn.Module): Model that should be trained
        train_data (torch.utils.Data.DataLoader): Data used for training
        val_data (torch.utils.Data.DataLoader): Data used for validating the training process
        num_epochs (int): amount of epochs done in this training
        optimizer (torch.optim): Optimizer that should be used for optimizing the model
        loss_function (torch.nn): Function that should be used for calculating the loss
        device (toch.device): GPU device on which the model should run
        path_to_save_model_to (str): Describes the path where the best model version should be saved to, e.g. cute-cats/Models/RobertaModel

    Returns:
        (list[float],list[float],list[float],list[float],list[float],list[float]): accuracies during training, fscores during training, losses during training, accuracies during validation, fscores during training, lossses during validation
    """
    print("start training")
    max_acc = 0
    max_fscore = 0
    train_epoch_accs, train_epoch_fscore, train_epoch_losses, val_epoch_accs, val_epoch_fscore, val_epoch_losses = [],[],[],[],[],[]

    # iterate num_epochs times over training data
    for epoch in range(num_epochs):
        print("Epoch ",epoch+1)

        # train the model on the train split
        tr_ep_acc, tr_ep_loss, tr_ep_fscore = train_epoch(model,train_data,optimizer,loss_function,device)
        print("\ttrain: \t",tr_ep_loss," (loss)\t",tr_ep_acc," (acc)\t", tr_ep_fscore," (F1-Score)")
        train_epoch_accs.append(tr_ep_acc)
        train_epoch_fscore.append(tr_ep_fscore)
        train_epoch_losses.append(tr_ep_loss)

        # at the same time evaluate the model on the validation split
        val_ep_acc, val_ep_loss, val_ep_fscore = validate_epoch(model,val_data,loss_function,device)
        print("\tvalidate: \t",val_ep_loss," (loss)\t",val_ep_acc," (acc)\t", val_ep_fscore," (F1-Score)")
        val_epoch_accs.append(val_ep_acc)
        val_epoch_fscore.append(val_ep_fscore)
        val_epoch_losses.append(val_ep_loss)

        # save model if it has a better accuracy or better F-Score
        if acc_bool:
            if val_ep_acc > max_acc:
                max_acc = val_ep_acc
                torch.save(model.state_dict(),path_to_save_model_to)
                print("\tSaved the model with accuracy: ", max_acc)
        else:
            if val_ep_fscore > max_fscore:
                max_fscore = val_ep_fscore
                torch.save(model.state_dict(),path_to_save_model_to)
                print("\tSaved the model with F1-Score: ", max_fscore)

    if acc_bool:
        print("Finished training, best model had an accuracy of ",max_acc)

    else:
        print("Finished training, best model had an F1-Score of ",max_fscore)

    plot_training(train_epoch_accs,train_epoch_losses,val_epoch_accs,val_epoch_losses,path_to_save_model_to+"_acc.png", "Accuracy")
    plot_training(train_epoch_fscore,train_epoch_losses,val_epoch_fscore,val_epoch_losses,path_to_save_model_to+"_fscore.png", "F1-Score")
    return train_epoch_accs, train_epoch_fscore,train_epoch_losses, val_epoch_accs, val_epoch_fscore, val_epoch_losses


def update_logging(logging_file_path,train_epoch_accs, train_epoch_fscores,train_epoch_losses, val_epoch_accs, val_epoch_fscores,val_epoch_losses):
    """Updates/Creates a file to keep track of influence of hyperparameter setting on training of our model(s)

    Args:
        logging_file_path (str): Path to a file where we log the training progress / influence of hyperparameters
        train_epoch_accs (list): List of accuracies when applying model on train data over the epochs
        train_epoch_fscores (list): List of fscores when applying model on train data on train over the epochs
        train_epoch_losses (list): List of losses when applying model on train data on train over the epochs
        val_epoch_accs (list): List of accuracies when applying model on validation data over the epochs
        val_epoch_fscores (list): List of fscores when applying model on validation data over the epochs
        val_epoch_losses (list): List of losses when applying model on validation data over the epochs
    """
    if not os.path.exists(logging_file_path):
        with open(logging_file_path,"w") as outfile:
            outfile.write("| Data | Hyperparams | Train-result | Val-result | ID |\n|----------------------|--------------------------------------|------------------|------------------|----|\n")
    with open(logging_file_path,"a") as outfile:
        if acc_bool:
            trained_for = "accuracy"
            ind_epoch_with_max = val_epoch_accs.index(max(val_epoch_accs)) 
        else:
            trained_for = "F1-score"
            ind_epoch_with_max = val_epoch_fscores.index(max(val_epoch_fscores))
        outfile.write("| Train: "+train_identifier+" | classification_layer_size = "+str(classification_layer_size) + " | Loss: "+str(round(train_epoch_losses[ind_epoch_with_max],4))+" | Loss: "+str(round(val_epoch_losses[ind_epoch_with_max],4)) + " | "+str(training_id) + "|\n" )
        outfile.write("| Dev: "+dev_identifier+" | num_epochs = "+str(num_epochs)+" | Accuracy: "+str(round(train_epoch_accs[ind_epoch_with_max],4))+" | Accuracy: "+str(round(val_epoch_accs[ind_epoch_with_max],4)) + "| saved: ep"+str(ind_epoch_with_max+1)+ "|\n")
        outfile.write("| | batch_size = "+str(batch_size)+" | F1-score: "+str(round(train_epoch_fscores[ind_epoch_with_max],4))+" | F1-score: "+str(round(val_epoch_fscores[ind_epoch_with_max],4)) + " | | \n")
        outfile.write("| | learning_rate = "+str(learning_rate)+" | | | |\n")
        outfile.write("| | loss_function = "+str(loss_function)+" | | | |\n")
        outfile.write("| | activation_function = "+str(activation)+" | | | |\n")
        outfile.write("| | optimizer = "+str(type(optimizer).__name__)+" | | | |\n")
        outfile.write("| | seed = "+str(seed)+" | | | |\n")
        outfile.write("| | optimized for = "+trained_for+" | | | |\n")
        outfile.write("|----------------------|--------------------------------------|------------------|------------------|----|\n")


# -----------------------------------------------------------------

# now actually train the model (evaluation on best model is included)
train_epoch_accs, train_epoch_fscores, train_epoch_losses, val_epoch_accs, val_epoch_fscores, val_epoch_losses = train(model,train_dataloader,dev_dataloader,num_epochs,optimizer,loss_function,device,saving_path)
update_logging(logging_file,train_epoch_accs, train_epoch_fscores, train_epoch_losses, val_epoch_accs, val_epoch_fscores, val_epoch_losses)