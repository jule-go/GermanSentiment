""" Author: Jule Godbersen (mailto:godbersj@tcd.ie)
    Content of file: Definition of model architecture"""

from torch import nn
import torch
import adapters
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig 
from adapters import AutoAdapterModel
from adapters import AdapterType
from adapters import LoRAConfig
from adapters import BnConfig
from adapters import AdapterConfig
from adapters import XLMRobertaAdapterModel
from opendelta import LoraModel, AdapterModel,AutoDeltaConfig,AutoDeltaModel
from peft import get_peft_model, LoraConfig, TaskType

class Classifier(nn.Module):
    """A class consisting of a pretrained xlm-roberta-base model architecture, adapter layers included, and a simple classification network.
    """
    def __init__(self,adapter_config,layer_size,activation,pretrained_model,device):
        """Constructs all the necessery attributes for this entire model, consisting of loading a roberta model, applying adapter on it and initiating a classification network

        Args:
            adapter_config (str): Underlying configuration of the adapter, e.g. "lora"
            layer_size (int): Size of the intermediate linear layer of the classification network
            activation (torch.nn): Activation function used in simple classification network, e.g. nn.ReLU()
            pretrained_model (str): name of pretrained model configuration
            device (torch.device): GPU device on which the model should run
        """
        super(Classifier,self).__init__()
        
        self.device = device

        # load pretrained model
        self.pretrained_model_name = pretrained_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.roberta_model = AutoModel.from_pretrained(self.pretrained_model_name) 
        self.roberta_model = self.roberta_model.to(device)

        if adapter_config == "none": # only train classification network and freeze roberta parameters
            for param in self.roberta_model.parameters():
                param.requires_grad = False

        # integrate adapter model

        elif adapter_config == "peft_lora":
            peft_config = LoraConfig(inference_mode=False,r=8,lora_alpha=32,lora_dropout=0.1) 
            self.roberta_model = get_peft_model(self.roberta_model,peft_config)
            self.roberta_model = self.roberta_model.to(device)
            self.roberta_model.print_trainable_parameters() # print amount of trainable parameters

        elif adapter_config == "opendelta_lora": # problem this model is not available for roberta-xlm
            delta_model = LoraModel(self.roberta_model)
            delta_config = AutoDeltaConfig.from_dict({"delta_type":"lora"})
            delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=self.roberta_model)
            delta_model = delta_model.to(device)
            delta_model.freeze_module(exclude=["deltas","classifier"])
        
        elif adapter_config == "opendelta": # problem this model doesn't learn
            delta_model = AdapterModel(backbone_model=self.roberta_model, modified_modules=['fc2'], bottleneck_dim=12)
            delta_model = delta_model.to(device)
            delta_model.freeze_module(exclude=["deltas","classifier"])
        
        elif adapter_config == "adapterhub_lora": # problem: doesn't learn
            self.self_pretrained_adapter_name = "xlm-roberta-base_adapter"
            adapters.init(self.roberta_model)
            ad_config = LoRAConfig(r=8,alpha=16)
            self.roberta_model.add_adapter(self.self_pretrained_adapter_name,config=ad_config,set_active=True)
            self.roberta_model.train_adapter(self.self_pretrained_adapter_name)
            # self.roberta_model.set_active_adapters(self.self_pretrained_adapter_name)
        
        elif adapter_config == "adapterhub_bn_seq": # problem: doesn't learn
            self.self_pretrained_adapter_name = "roberta-base_adapter"#"xlm-roberta-base_adapter"
            adapters.init(self.roberta_model)
            ad_config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
            self.roberta_model.add_adapter(self.self_pretrained_adapter_name,config=ad_config,set_active=True)
            self.roberta_model.train_adapter(self.self_pretrained_adapter_name)
            # self.roberta_model.set_active_adapters(self.self_pretrained_adapter_name)

        elif adapter_config == "otherApproach": # try loading a pretrained adapter -> still didn't learn
            adapters.init(self.roberta_model)
            adapter_name = self.roberta_model.load_adapter("@ukp/xlm-roberta-base-en-wiki_pfeiffer",config="pfeiffer")# found via from adapters import list_adapters, adapter_infos = list_adapters(source="ah",model_name="xlm-roberta-base")
            self.roberta_model.train_adapter(adapter_name)
            self.roberta_model.set_active_adapters(adapter_name)


        # define simple classification network
        self.class_layer1 = torch.nn.Linear(768,layer_size) # iput is roberta output of size 768, output is layer_size
        self.class_layer2 = torch.nn.Linear(layer_size,3) # use 3 as we have three classes (neg, neut, pos)
        self.activation = activation
        self.softmax = nn.Softmax(dim=1) # directly get probability distribution via softmax


    def forward(self,texts): # texts is a list of strings for which we want to have the sentiment
        """Represents the call of the Classifier model, making the sentiment prediction for a given batch of texts

        Args:
            texts (list): Input texts

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

        return predictions