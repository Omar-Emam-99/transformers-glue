"""Test Train Model on a test dataset"""
import torch 
import evaluate
from torch.utils.data import DataLoader
import transformers
from transformers import BertForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer , BertForSequenceClassification
from transformers import TextClassificationPipeline
from typing import Text
import pandas as pd

class Test :
    
    def __init__(self , configs):
        """
        arg :
            configuration file that contains information about :
                test_data preprocessed path
                Trained Model path
            
        """
        self.test_data = configs['tokenized_data']
        self.saved_model_path = configs['trainer']
        self.predict_outs_dir = configs["tester"]["predicionts_dir"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    def get_dataloader(self):
        """load tokenized test data and do some processes on it and convert it to batches"""
        test_data = load_dataset('json' , data_files=f"{self.test_data['path']}/test.json")
        test_data = test_data.rename_column("label" , "labels")
        test_data = test_data.remove_columns(['text'])
        test_data.set_format("torch")
        test_dataload = DataLoader(test_data['train'] , batch_size=64)
        
        return test_dataload
    
    def get_model(self):
        """bring saved model"""
        model_ckpt = f"{self.saved_model_path['out_dir']}"
        model = BertForSequenceClassification.from_pretrained(model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        return model , tokenizer
        
    def test(self):
        model , _ = self.get_model()
        model.to(self.device)
        metric = evaluate.load("accuracy")
        for batch in self.get_dataloader():
            batch = {k , v.to(self.device) if k , v in batch.items()}
            with torch.zero_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits , dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        accuracy = metric.compute()

        print(f"Test data accuracy : {accuracy}.")
    
    def predict(self , input_text : Text):
        """predict any new review sentance"""
        model , tokenizer = self.get_model()
        classifier = TextClassificationPipeline(model=model ,tokenizer=tokenizer ,return_all_scores=True)
        outs = classifier(input_text)
        pd.DataFrame(outs).to_csv(f"{self.predict_outs_dir}")
        print(f"Outputs Scores : {outs[0]}")
        
        
        
        
        
