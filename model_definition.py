""" Author: Jule Godbersen (mailto:godbersj@tcd.ie)
    Content of file: Definition of model architecture"""

from torch import nn
import torch
import adapters
from transformers import AutoTokenizer
from transformers import AutoModel

class Classifier(nn.Module):
    """A class consisting of a pretrained xlm-roberta-base model architecture, adapter layers included, and a simple classification network.
    """
    def __init__(self,adapter_config,layer_size,activation,device):
        """Constructs all the necessery attributes for this entire model, consisting of loading a roberta model, applying adapter on it and initiating a classification network

        Args:
            adapter_config (str): Underlying configuration of the adapter, e.g. "lora"
            layer_size (int): Size of the intermediate linear layer of the classification network
            activation (torch.nn): Activation function used in simple classification network, e.g. nn.ReLU()
            device (torch.device): GPU device on which the model should run
        """
        super(Classifier,self).__init__()
        
        self.device = device

        # load pretrained model
        self.pretrained_model_name = "xlm-roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.roberta_model = AutoModel.from_pretrained(self.pretrained_model_name) 
        self.roberta_model = self.roberta_model.to(device)

        # integrate adapter model
        if adapter_config != "none":
            self.self_pretrained_adapter_name = "xlm-roberta-base_adapter"
            adapters.init(self.roberta_model)
            self.roberta_model.add_adapter(self.self_pretrained_adapter_name,config=adapter_config)
            self.roberta_model.train_adapter(self.self_pretrained_adapter_name)

        # define simple classification network
        self.class_layer1 = torch.nn.Linear(768,layer_size) # iput is roberta output of size 768, output is layer_size
        self.class_layer2 = torch.nn.Linear(layer_size,3) # use 3 as we have three classes (neg, neut, pos)
        self.activation = activation
        self.softmax = nn.Softmax(dim=1) # directly get probability distribution via softmax


    def forward(self,texts,labels,loss_function): # texts is a list of strings for which we want to have the sentiment
        """Represents the call of the Classifier model, making the sentiment prediction for a given batch of texts

        Args:
            texts (list): Input texts
            labels (torch.tensor): One-hot encoding of prediction, e.g. torch.tensor([0,0,1]) with index 0 representing negative, index 1 representing neutral and index 2 representing positive
            loss_function (torch.nn): Function that should be used for calculating the loss

        Returns:
            (torch.tensor): Probability distribution over classes as predicted with help of the roberta embeddings, adapter layers, and the classification network
        """

        # prepare texts by tokenizing them
        tokenized_texts = self.tokenizer(texts,padding=True,truncation=True,return_tensors="pt")
        input_ids = tokenized_texts["input_ids"].to(self.device)
        attention_mask = tokenized_texts["attention_mask"].to(self.device)

        # apply xlm-roberta model 
        roberta_output = self.roberta_model(**{"input_ids":input_ids,"attention_mask":attention_mask})
        cls_embedding_pooled = roberta_output.pooler_output # is tensor of shape [batch_size, 768]; cf. "Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function." (see e.g. https://huggingface.co/transformers/v3.0.2/model_doc/roberta.html) 

        # apply classification network 
        intermediate_prediction = self.class_layer1(cls_embedding_pooled)
        activated_prediction = self.activation(intermediate_prediction)
        predictions = self.class_layer2(activated_prediction)

        # evtl. directly calculate the loss
        if labels != None:
            loss = loss_function(predictions, labels)
            return loss, predictions
        else:
            return predictions