import pandas as pd
import pytreebank , os , sys

#Read and convert data from (TREEBANK) format to(CSV) file 
def convert_data_from_treebank_to_text(file_path):
    #load 3 data files
    data = pytreebank.load_sst(path)
    #output new formation 
    out_path = os.path.join("./data/trainDevTestTrees_PTB/sst_{}.txt")
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
    
    #create dictionary to contain a [train-dev-test].csv path
    dataset = dict()
    for data in ["train" , "dev" , "test"]:
        #read three dataset file
        file = pd.read_csv(f"./data/trainDevTestTrees_PTB/{data}.txt" ,sep="\t",header=None,
                           names=['label','text'])
        #replace __label__ with space and convevrt it to int type
        file["label"] = pd.to_numeric(file["label"].str.replace("__label__" , ""))
        #apply sub to get label range from[0-4] instead of [1-5]
        file["label"] = file["label"].apply(lambda x : x - 1) 
        #change the pos of columns
        file=file[["text" , "label"]]
        dataset[data]= file
    #create CSV file for Train , dev and test sets
    for i in dataset:
        dataset[i].to_csv(f"../data/Cleaned_data/{i}.csv" , index=False)