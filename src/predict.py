#!/usr/bin/env python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging

logging.set_verbosity_error()

class MyModel:
    
    tokenzier = None
    model = None
    device = None

    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small").to(device)
        self.device = device

    def load_test_data(self, fname):
        # your code here
        data = []
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    def write_pred(self, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                  f.write('{}\n'.format(p))

    def run_pred(self, data):
        preds = []

        if self.tokenizer is None or self.model is None:
            print("Please load the model first")
            return preds

        self.model.eval()
        for inp in tqdm(data):
            results = []
            try:
                input_ids = self.tokenizer(inp).input_ids
                input_ids = torch.tensor([input_ids[:-1] + [258] + [1]]).to(self.device)
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
                print(f"Error when testing for {inp}: {e}")
                results = ['a', 'e', 'i']

            preds.append(''.join(results))

        return preds


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    print('Loading model')
    model = MyModel()
    print('Loading test data from {}'.format(args.test_data))
    test_data = model.load_test_data(args.test_data)
    print('Making predictions')
    pred = model.run_pred(test_data)
    print('Writing predictions to {}'.format(args.test_output))
    if len(pred) != len(test_data):
        print(f'Warning: number of predictions {len(pred)} does not match number of test data {len(test_data)}')
        if len(pred) > len(test_data):
            pred = pred[:len(test_data)]
        else:
            pred.extend(['aei'] * (len(test_data) - len(pred)))
    
    model.write_pred(pred, args.test_output)
    print("Done! Predictions saved to", args.test_output)
