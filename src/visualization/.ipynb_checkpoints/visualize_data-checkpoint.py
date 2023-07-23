import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import sys
src_dir = Path.cwd()
sys.path.append(str(src_dir))
from src.stages.load_data import load_datasets 



class Visualization :
    def __init__(self , dataset):
        self.dataset = dataset
        self.df = dataset.set_format(type="pandas")
    
    def visualize_class_frequency(self):

        def label_int_toStr(row):
          return self.dataset['train'].features["label"].int2str(row)
        self.df = self.dataset['train'][:]
        self.df["label_name"] = self.df['label'].apply(label_int_toStr)
        self.df["label_name"].value_counts(ascending=True).plot.barh()
        plt.title("Frequency Classes")
        plt.show()

        
    def visualize_len_of_sentance(self):
        self.df["Words per sentance"] = self.df['text'].str.split().apply(len)
        self.df.boxplot("Words per sentance", by="label_name" ,grid=False ,showfliers=True , color="red")
        plt.suptitle("")
        plt.ylabel("lens of sentance")
        plt.show()


if __name__ == '__main__':
    with open("param.yaml") as config_file :
        config = yaml.safe_load(config_file)
    rotten_movie = load_datasets(config["load_data"]["path"])
    #visualize_class_frequency(rotten_movie)
    Vis = Visualization(rotten_movie)
    Vis.visualize_class_frequency()
    Vis.visualize_len_of_sentance()