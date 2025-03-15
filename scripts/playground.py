import os
from datasets import load_dataset
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, logging

def main():
    train_dir = "data/train"
    val_dir = "data/val"
    data_files = {
        "train": [os.path.join(train_dir, f) for f in os.listdir(train_dir)],
        "validation": [os.path.join(val_dir, f) for f in os.listdir(val_dir)]
    }
    
    extension = "text"
    
    print(f"Loading dataset from {train_dir} and {val_dir}")
    datasets = load_dataset(
        extension,
        data_files=data_files,
        num_proc=4,
    )
    print(datasets['validation'])
    print("Finished loading dataset")

if __name__ == '__main__':
    main()
