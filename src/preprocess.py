import os
import json
import torch
from transformers import AutoTokenizer, MarianMTModel, MarianTokenizer
from tqdm import tqdm

# Define paths
RAW_DATA_DIR = "data/raw"  # Directory containing raw data
PROCESSED_DATA_DIR = "data/processed"  # Directory for saving processed data
LANGUAGES = ["fr", "de", "es", "zh"]  # Languages to translate into

# Load XLM-R tokenizer
xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# MarianMT translation models
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

if __name__ == "__main__":
    print("Loading raw data...")
    texts = load_raw_data()
    
    print("Translating texts to multiple languages...")
    all_translations = texts.copy()  # Keep original English texts
    for lang in LANGUAGES:
        all_translations.extend([translate_text(text, lang) for text in tqdm(texts, desc=f"Translating to {lang}")])

    print("Tokenizing and generating sequences...")
    tokenized_data = preprocess_texts(all_translations)
    
    print("Saving processed data...")
    save_processed_data(tokenized_data, "training_data.json")

    print("Preprocessing complete! Data saved in 'data/processed/training_data.json'")
