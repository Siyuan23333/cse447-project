#!/usr/bin/env python
import os
import string
import random
import json
import torch
from transformers import AutoTokenizer, MarianMTModel, MarianTokenizer
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    RAW_DATA_DIR = "data/raw"  # Directory containing raw data
    PROCESSED_DATA_DIR = "data/processed"  # Directory for saving processed data
    LANGUAGES = ["fr", "de", "es", "zh"]  # Languages to translate into

    translation_models = {
        lang: MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
        for lang in LANGUAGES
    }
    translation_tokenizers = {
        lang: MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
        for lang in LANGUAGES
    }

    def load_raw_data():
        """ Loads raw NASA astronaut text for preprocessing. """
        all_texts = []
        for filename in os.listdir(RAW_DATA_DIR):
            with open(os.path.join(RAW_DATA_DIR, filename), "r", encoding="utf-8") as f:
                all_texts.extend(f.readlines())
        return [line.strip() for line in all_texts if line.strip()]
    
    def translate_text(text, target_lang):
        """ Translates a given English text to the target language. """
        tokenizer = translation_tokenizers[target_lang]
        model = translation_models[target_lang]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)
        
        return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    def preprocess_texts(texts):
        """ Tokenizes and creates character sequences for training. """
        tokenized_texts = []
        seq_length = 20  # Length of input sequence
    
        for text in tqdm(texts, desc="Tokenizing texts"):
            tokens = xlmr_tokenizer.encode(text, add_special_tokens=True)
            
            # Create sliding window sequences
            for i in range(len(tokens) - seq_length):
                input_seq = tokens[i : i + seq_length]
                target_char = tokens[i + seq_length]
                tokenized_texts.append((input_seq, target_char))
    
        return tokenized_texts
    
    def save_processed_data(data, filename):
        """ Saves processed data as JSON. """
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        with open(os.path.join(PROCESSED_DATA_DIR, filename), "w") as f:
            json.dump(data, f)


    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        pass

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
