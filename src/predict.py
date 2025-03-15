import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, logging

MASK_TOKEN_ID = 258

def load_test_data(txt_file):
    data = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.rstrip("\n"):
                data.append(line.rstrip("\n") + "m")
    print(len(data), "lines in test data")
    return data

def prepare_test_data(tokenizer, data, mask_token_id=MASK_TOKEN_ID):
    tokenized_data = tokenizer(data, padding=True, truncation=True, max_length=64, return_tensors="pt")
    true_length = torch.sum(tokenized_data['attention_mask'], dim=1) - 1
    batch_indices = torch.arange(tokenized_data["input_ids"].shape[0])
    tokenized_data["input_ids"][batch_indices, true_length - 1] = mask_token_id
    return tokenized_data

def get_char_predictions(decoded_results, num_return_sequences=3):
    char_predictions = []
    for i in range(len(decoded_results) // num_return_sequences):
        char_prediction = ""
        for j in range(num_return_sequences):
            pred = decoded_results[i * num_return_sequences + j]
            if pred == "":
                char_prediction += " "
            else:
                char_prediction += pred[0] if pred[0] != "\n" else " "
        char_predictions.append(char_prediction)
    return char_predictions

def predict(model, tokenizer, test_data, batch_size=16, num_return_sequences=3):
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            if i + batch_size > len(test_data):
                batch = test_data[i:]
            else:
                batch = test_data[i:i + batch_size]
            try:
                tokenized_testdata = prepare_test_data(tokenizer, batch)
                tokenized_testdata = {
                    key: value.to(device) for key, value in tokenized_testdata.items()
                }
                output = model.generate(
                    **tokenized_testdata,
                    max_new_tokens=7,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    top_k=5,
                    temperature=1.5,
                )
                decoded_results = tokenizer.batch_decode(output, skip_special_tokens=True)
                char_predictions = get_char_predictions(decoded_results, num_return_sequences)
                results.extend(char_predictions)
            except Exception as e:
                char_predictions = ["ae "] * len(batch)
                results.extend(char_predictions)
                
    return results

def write_pred(preds, fname):
    with open(fname, 'wt') as f:
        for p in preds:
            f.write('{}\n'.format(p))

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()
    print(f"Loading test data from {args.test_data}")
    test_data = load_test_data(args.test_data)
    
    try:
        file_path = os.path.abspath(__file__)
        model_name = os.path.join(os.path.dirname(file_path), "../work")
        print(f'Loading model and tokenizer from {model_name}')
        logging.set_verbosity_error()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.truncation_side = "left"
        
        print(f"Predicting on test data")
        batch_size = 32
        num_return_sequences = 3
        pred = predict(model, tokenizer, test_data, batch_size=batch_size, num_return_sequences=num_return_sequences)

        if len(pred) != len(test_data):
            if len(pred) > len(test_data):
                pred = pred[:len(test_data)]
            else:
                pred.extend(['ae '] * (len(test_data) - len(pred)))
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        pred = ["ae "] * len(test_data)
        
    write_pred(pred, args.test_output)
    print("Done! Predictions saved to", args.test_output)