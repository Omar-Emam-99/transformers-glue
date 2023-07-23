from transformers import AutoTokenizer
import datasets , yaml
import argparse
from typing import Text
import pytreebank , os , sys



def tokenize_data(path : Text) -> None :
    with open(path) as config :
        path = yaml.safe_load(config)
    data = dict()    
    for i in ["train" , "dev" , "test"] : 
        data[i] = f"{path['load_data']['path']}{i}.csv"
    dataset = datasets.load_dataset('csv' , data_files = data)
    #create tokenizer form DISTILBERT model
    model_ckp = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckp)
    #apply tokenizet
    def tokenization(batch):
        return tokenizer(batch['text'] , padding=True , truncation=True)
    #map tokenization function to all dataset
    dataset_encoded = dataset.map(tokenization, batched=True , batch_size=None)
    print(dataset_encoded)
    #create json files to save our tokenization step 
    for split , dataset in dataset_encoded.items() :
        dataset.to_json(f"./src/data/{split}.json")


if __name__ == '__main__':
    
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config' , dest='config' , required=True)
    args =  arg_parse.parse_args()
    tokenize_data(path=args.config)
    