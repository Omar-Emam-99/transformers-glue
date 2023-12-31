"""Train SequenceClassifierModel"""
import torch
import evaluate
import pandas as pd
import numpy as np
from typing import Text
from pathlib import Path
from tqdm.auto import tqdm
import argparse , sys , json
from datasets import load_dataset
from collections import defaultdict
from src.models.modeling_bert import *
from torch.optim import AdamW , Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler , AutoConfig
import functools , operator
from sklearn.metrics import confusion_matrix
from transformers.modeling_outputs import SequenceClassifierOutput
#src_dir = Path.cwd()
#sys.path.append(str(src_dir))


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
        self.labels = config["trainer"]["binary_labels"] if config["trainer"]["num_labels"] == 2 \
        else config["trainer"]["three_class_labels"] if config["trainer"]["num_labels"] == 3 else \
        config["trainer"]["labels"]
        
        
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
                                   id2label={i:k for i,k in enumerate(self.labels)},
                                   label2id={k:i for i,k in enumerate(self.labels)})

        return (BertForSequenceClassification.from_pretrained(model_ckpt,config=config))
    
    
    def train(self):
        """train the model, return losses , accuracy and save it"""
        #Get data and convet it to batches
        train_data = self.get_dataloader(self.data_args['path'])
        eval_data = self.get_dataloader(self.data_args['path']) 
        train_dataloader = DataLoader(train_data["train"],
                                      shuffle=True,
                                      batch_size =self.train_args['batch_size'])
        
        eval_dataloader = DataLoader(eval_data["dev"],
                                     batch_size = self.train_args['batch_size'])
        
        num_of_train_data = len(train_dataloader)
        
        #initialize a learning rate schaduler
        num_training_steps = self.train_args['num_epochs'] * num_of_train_data
        
        losses = defaultdict(list)
        progress_bar = tqdm(range(num_training_steps), colour="green")
        
        #initialize the model
        model = self.init_model()
        model.to(device)
        
        #accuracy
        accuracy_metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        #Confusion Metrix 
        cfm_metric = evaluate.load("BucketHeadP65/confusion_matrix")
        preds_labels = defaultdict()
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
                
                
                accuracy_metric.add_batch(predictions=predictions,references=batch["labels"])
                losses['preds'].append(predictions.cpu().numpy())
                losses['labels'].append(batch['labels'].cpu().numpy())
                
            valid_metrics = {"valid_{}".format(k):v for k,v in accuracy_metric.compute().items()}
            losses["metrics"].append(valid_metrics)
            print("---------------------------------------------------------------------------")
            print(f"\nTraining loss : {loss}\nEval loss : {valid_loss}\n{valid_metrics}")
            print("---------------------------------------------------------------------------") 
        

        labels_cm = np.concatenate(losses['labels'])
        pred_cm = np.concatenate(losses['preds'])
        cm = confusion_matrix(y_true=labels_cm ,y_pred=pred_cm)
        #print(cm)
        
        with open(os.path.join(self.train_args["metrics_path"],"vaild_metrics.json"),'w') as file :
            json.dump(valid_metrics,file)
        
        #store training losses per batches to plot them    
        pd.DataFrame({"batches": list(range(len(losses['train']))) ,
                      "train_loss":losses['train']})\
        .to_csv(os.path.join(self.train_args["reports_dir"], "train_loss.csv"), index=False)
        #store validation losses per batches to plot them
        pd.DataFrame({"batches": list(range(len(losses['eval']))) ,
                      "valid_loss":losses['eval']})\
        .to_csv(os.path.join(self.train_args["reports_dir"], "eval_loss.csv"), index=False)
        
        #save Confusion Metrics to csv file to plot
        pd.DataFrame(cm ,columns=self.labels,index=self.labels).to_csv(os.path.join(self.train_args["reports_dir"],
                                                                                 "confusion_metrix.csv"))
        
        
            #Save model wieghts and configurations
        model.save_pretrained(self.train_args["out_dir"])