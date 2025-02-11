#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
import torch

logging.set_verbosity_error()

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    tokenzier = None
    model = None

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

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        return

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file

        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
        
        instance = cls()
        instance.tokenizer = tokenizer
        instance.model = model

        return instance


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
