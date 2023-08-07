"""Train SequenceClassifierModel"""
from pathlib import Path
import torch
import sys
import yaml
src_dir = Path.cwd()
sys.path.append(str(src_dir))
import argparse
from src.models.Trainer import *

if __name__ == '__main__' :
    
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config' , dest='config' , required=True)
    args =  arg_parse.parse_args()
    with open(args.config) as obj :
        config_data = yaml.safe_load(obj)

    trainer = Trainer(config_data)
    trainer.train()