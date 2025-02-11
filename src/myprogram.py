#!/usr/bin/env python
import os
import string
import random
import json
import torch
from transformers import AutoTokenizer, MarianMTModel, MarianTokenizer
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
import torch

logging.set_verbosity_error()

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
<<<<<<< HEAD
    tokenzier = None
    model = None
=======
    def __init__(self, model_name="xlm-roberta-base", work_dir="work"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)

        # Load trained classifier
        self.hidden_size = self.encoder.config.hidden_size
        self.classifier = torch.nn.Linear(self.hidden_size, len(string.ascii_letters) + 10).to(self.device)

        model_path = os.path.join(work_dir, "xlmr_character_model.pt")
        if os.path.exists(model_path):
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Loaded trained model.")
        else:
            print("Warning: No trained model found. Using random predictions.")
>>>>>>> 6641f10192ac9762b30d295f54381251edbcdee7

    @classmethod
    def load_training_data(cls):
        # your code here
        # Training should be done in train.py
        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        # data = []
        # with open(fname) as f:
        #     for line in f:
        #         inp = line[:-1]  # the last character is a newline
        #         data.append(inp)
        # return data

        with open(fname, "r") as f:
            return [line.strip() for line in f]

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def predict_next_char(self, text):
        """ Predicts next characters using trained XLM-R model. """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)

        hidden_states = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(hidden_states)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_indices = torch.topk(probs, 3, dim=-1).indices.cpu().numpy().flatten()
        top_chars = [string.ascii_letters[i % len(string.ascii_letters)] for i in top_indices]

        return "".join(top_chars)  # Return top 3 predicted characters

    def run_train(self, data, work_dir):
        # your code here
        pass

    def run_pred(self, data):
        # your code here
<<<<<<< HEAD
        preds = []

        if self.tokenizer is None or self.model is None:
            print("Please load the model first")
            return preds

        self.model.eval()
        for inp in data:
            results = []
            try:
                input_ids = self.tokenizer(inp).input_ids
                input_ids = torch.tensor([input_ids[:-1] + [258] + [1]])
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=30,
                        num_beams=1,
                        do_sample=False,
                        num_return_sequences=1,
                        early_stopping=False,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                scores = outputs.scores[1][0].clone()
                scores[self.tokenizer.all_special_ids] = -float("inf")
                _, indices = torch.topk(scores, k=3)
                results = [self.tokenizer.decode(ind) for ind in indices]
            except Exception as e:
                print(f"Error when testing: {e}")

            preds.append(''.join(results))

        return preds
=======
        # preds = []
        # all_chars = string.ascii_letters
        # for inp in data:
        #     # this model just predicts a random character each time
        #     top_guesses = [random.choice(all_chars) for _ in range(3)]
        #     preds.append(''.join(top_guesses))
        # return preds
        return [self.predict_next_char(text) for text in data]
>>>>>>> 6641f10192ac9762b30d295f54381251edbcdee7

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
<<<<<<< HEAD
        return
=======
        # Model is saved in train.py
        
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
        #     f.write('dummy save')
        pass
>>>>>>> 6641f10192ac9762b30d295f54381251edbcdee7

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
<<<<<<< HEAD

        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
        
        instance = cls()
        instance.tokenizer = tokenizer
        instance.model = model

        return instance
=======
        # with open(os.path.join(work_dir, 'model.checkpoint')) as f:
        #     dummy_save = f.read()
        # return MyModel()
        return cls(work_dir=work_dir)
>>>>>>> 6641f10192ac9762b30d295f54381251edbcdee7


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
        print("Done! Predictions saved to", args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
