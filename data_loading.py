""" Author: Jule Godbersen (mailto:godbersj@tcd.ie)
    Content of file: Loading the dataset (contents) that are later used as input for the model"""

from torch.utils.data import Dataset, DataLoader


def load_own_dataset(data, data_ids, translations=None) -> Dataset:
    """Loads sentiment analysis dataset
    Args:
        data (Huggingface dataset): Underlying dataset of this project
        data_ids (list): List of ids that represent instances of this datasplit
        translations (dict): keys are ids of data instance, values are the according translations
    Returns:
        Dataset: Sentiment analysis data (file content) in correct format
    """
    return SentimentAnalysisData(data,data_ids,translations)


def own_collate(batch) -> dict:   
    """Define how the Dataloader should collate the datapoints
    Args:
        batch: Batch of datapoints of dataset
    Returns:
        dict: Content of batch as dict with lists of datapoint-contents as values
    """    
    texts = []
    labels = []
    ids = []
    for datapoint in batch:
        texts += [datapoint["text"]]
        labels += [datapoint["label"]]
        ids += [datapoint["id"]]
    return {"texts":texts,"labels":labels,"ids":ids}


def load_dataloader(dataset,batch_size) -> DataLoader:
    """Creates a Dataloader from an existing dataset with the specified batch size. 
       Data samples are being shuffled and we only consider full batches
    Args:
        dataset (Dataset): Is the loaded dataset
        batch_size (int): Determines how large the batches should be
    Returns:
        DataLoader: Dataloader that is needed for using our models later
    """    
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,collate_fn=own_collate,drop_last=True)


class SentimentAnalysisData(Dataset):
    """A class to represent the sentiment analysis data
    Attributes:
        texts (list): texts that should be evaluated for their sentiment
        labels (list): binary plausible / implausible ratings
    """    
    def __init__(self,data,data_ids,translations=None):
        """ Constructs all the necessary attributes for the SentimentAnalysis class
        Args:
            data (Huggingface dataset): Underlying dataset of this project
            data_ids (list): List of ids that represent instances of this datasplit
            translations (dict): keys are ids of data instance, values are the according translations
        """ 
        self.texts = [] 
        self.labels = [] 
        self.ids = []

        for data_id in data_ids:   
            self.ids += [data_id]
            current_instance = data["train"][data_id]
            self.labels += [current_instance["label"]]
            if translations:
                self.texts += [translations[data_id]]
            else:
                self.texts += [current_instance["text"]]
    

    def __len__(self) -> int:
        """Outputs the amount of datapoints within this dataset
        Returns:
            (int): Length of dataset
        """        
        return len(self.labels)
    

    def __getitem__(self, index) -> dict:
        """Allows accessing items of dataset via their index
        Args:
            index (int): Index of datapoint in dataset
        Returns:
            dict: Information, consisting of S-V-O triplet, plausibility-label and further information for specified datapoint
        """        
        text = self.texts[index]
        label = self.labels[index]
        id = self.ids[index]
        return {"text":text,"label":label,"id":id}