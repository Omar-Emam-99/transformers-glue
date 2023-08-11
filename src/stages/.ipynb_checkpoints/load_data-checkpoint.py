'''
This step to load the treeBank dataset and ingest it into transformers dataset
'''
import yaml
import datasets
import argparse
from typing import Text
import pandas as pd
import pytreebank , os , sys


label_dict = {3 :{
                1:0,
                2:0,
                3:1,
                4:2,
                5:2},
             2 :{
                1 : 0,
                2 : 0,
                3 : 1,
                4 : 1,
                5 : 1
             }}
def label_indetify(label, n_labels):
    if n_labels == 3:
        return label_dict[3][label]
    elif n_labels == 2:
        return label_dict[2][label]
    else : 
        return label - 1

def load_datasets(path : Text) -> None:
    with open(path) as con:
        config = yaml.safe_load(con)
    #load 3 data files
    data = pytreebank.load_sst(config['data']['path_tree'])
    #output new formation 
    out_path = os.path.join("data/trainDevTestTrees_PTB/trees/sst_{}.txt")
    #iterate on every dataset and convert the tree format to simple text
    for cat in ["train", "test" , "dev"]:
        with open(out_path.format(cat) , "w") as file :
            for item in data[cat]:
                file.write(
                    "{}\t{}\n".format(
                        item.to_labeled_lines()[0][0]+1, 
                        item.to_labeled_lines()[0][1]
                    )
                )
        print("done with {}".format(file))
    
    #create dictionary to contain a three file and convert them to CSV files
    dataset = dict()
    for data in ["train" , "dev" , "test"]:
        #read three dataset file
        file = pd.read_csv(f"{config['data']['path_tree']}/sst_{data}.txt" ,sep="\t",header=None,names=
                                                                                       ['label','text'])
        #replace __label__ with space and convevrt it to int type
        #file["label"] = pd.to_numeric(file["label"].str.replace("__label__" , ""))
        #apply sub to get label range from[0-4] instead of [1-5]
        file["label"] = file["label"].apply(lambda x : label_indetify(x, config["trainer"]["num_labels"])) 
        #change the pos of columns
        file=file[["text" , "label"]]
        dataset[data]= file
    print(dataset["train"]["label"])
    if not os.path.exists(config['load_data']['path']):
        os.mkdir(config['load_data']['path'])
    #create CSV file for Train , dev and test sets
    for path in dataset:
        dataset[path].to_csv(os.path.join(config['load_data']['path'],f'{path}.csv'), index=False)
        #dataset[i] = f"{config['load_data']['path']}/{i}.csv"
        print(f"{path} is saved in {config['load_data']['path']}......")
    

if __name__ == '__main__':
    # define argument pareser
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config' , dest='config' , required=True)
    args =  arg_parse.parse_args()
    load_datasets(path=args.config)