import yaml
import argparse
from typing import Text
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification ,TextClassificationPipeline

class Inference :
    def __init__(self , config):
        self.saved_model_path = configs['trainer']
       
    def get_model(self):
        """bring saved model"""
        model_ckpt = f"{self.saved_model_path['out_dir']}"
        model = BertForSequenceClassification.from_pretrained(model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        return model , tokenizer

    def predict(self , input_text : Text):
        """predict any new review sentance"""
        model , tokenizer = self.get_model()
        classifier = TextClassificationPipeline(model=model ,tokenizer=tokenizer ,top_k=1)
        outs = classifier(input_text)
        print(f"Predictions of Movie Reviews is :\nLabel : {outs[0][0]['label']}\nScore : {outs[0][0]['score']}")


if __name__ == "__main__" :
    
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--predict' , dest='predict' , required=True)
    args =  arg_parse.parse_args()
    
    #get configurations
    with open("params.yaml") as obj :
        configs = yaml.safe_load(obj)

    model = Inference(configs)
    model.predict(args.predict)