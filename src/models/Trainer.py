"""Train SequenceClassifierModel"""
from pathlib import Path
import argparse
import sys
import json
from tqdm.auto import tqdm
import evaluate
from datasets import load_dataset
from typing import Text
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW , Adam
from collections import defaultdict
from transformers import get_scheduler
from transformers import AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
src_dir = Path.cwd()
sys.path.append(str(src_dir))
from src.models.modeling_bert import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer :
    def __init__(self, config):
        
        """Trainer Class take a train and validation data and apply
        arg :
            train_parameters
                out_dir : A path directory to save model artifacts.
                num_labels : number of classification labels ,Int.
                batch_size : number of batches , default 64.
                num_epochs : number of epoches , default 2.
                learning_rate : learning rate for optimizer , default 5e-5. 
            data_params :
                data : A Path to tokenized data file in any format.
            """
        self.train_args = config['trainer']
        self.data_args = config['tokenize_data']
        
    
    def get_dataloader(self, data : Text):
        dataset = {}
        for data_type in ["train" , "dev"]:
            dataset[data_type] = f"{self.data_args['path']}/{data_type}.json"
        data = load_dataset('json' , data_files=dataset)
        data = data.rename_column("label" , "labels")
        data = data.remove_columns(['text'])
        data.set_format("torch")
        return data
    
    def init_model(self):
        """initialize the BertForSequenceClassification Model"""
        model_ckpt = "bert-base-uncased"
        config = AutoConfig.from_pretrained(model_ckpt,
                                   num_labels = self.train_args['num_labels'],
                                   id2label={i:k for i,k in enumerate(self.train_args['labels'])},
                                   label2id={k:i for i,k in enumerate(self.train_args['labels'])})

        return (BertForSequenceClassification.from_pretrained(model_ckpt,config=config))
    
    def train(self):
        """train the model, return losses , accuracy and save it"""
        #Get data and convet it to batches
        train_data = self.get_dataloader(self.data_args['path'])
        eval_data = self.get_dataloader(self.data_args['path']) 
        train_dataloader = DataLoader(train_data["train"],shuffle=True ,batch_size = self.train_args['batch_size'])
        eval_dataloader = DataLoader(eval_data["dev"], batch_size = self.train_args['batch_size'])
        
        num_of_train_data = len(train_dataloader)
        
        #initialize a learning rate schaduler
        num_training_steps = self.train_args['num_epochs'] * num_of_train_data
        
        losses = defaultdict(list)
        progress_bar = tqdm(range(num_training_steps))
        
        model = self.init_model()
        model.to(device)
        
        metric = evaluate.load("accuracy")
        
        optimizer = AdamW(model.parameters(), lr=self.train_args["learning_rate"])
        LR_Schaduler =get_scheduler(name='linear' ,
                                    optimizer= optimizer,
                                    num_warmup_steps = 0,
                                    num_training_steps= num_training_steps)
        
        for epoch in range(self.train_args["num_epochs"]):

            model.train()
            for batch in train_dataloader:

                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                loss = outputs.loss
                loss.backward()
                losses['train'].append(float(loss))
                optimizer.step()
                LR_Schaduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            model.eval()
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                valid_loss = outputs.loss
                losses['eval'].append(float(valid_loss))
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                #print(predictions)
                metric.add_batch(predictions=predictions, references=batch["labels"])
            acc = metric.compute()
            losses["accuracy"].append(acc) 
            print(f"\nTraining loss : {loss} ,Eval loss : {valid_loss} , Accuracy : {acc}")
        """
        with open("reports/losses.json",'w') as fp :
            json.dump(losses,fp)
        """
        pd.DataFrame({"train_loss":losses['train'],
                      "batches": list(range(len(losses['train'])))})\
        .to_csv(os.path.join(self.train_args["reports_dir"], "train_loss.csv"))
                    
        model.save_pretrained(self.train_args["out_dir"])