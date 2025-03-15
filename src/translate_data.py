import os
import glob
import time
import openai
import dotenv
import json
from tqdm import tqdm

dotenv.load_dotenv() 

openai.api_key = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY_HERE"

INPUT_DIR = "/Users/siyuange/Documents/CSE447/cse447-project/data/processed/en"
OUTPUT_DIR = "/Users/siyuange/Documents/CSE447/cse447-project/data/processed/zh"
TARGET_LANGUAGE = "Chinese"  
BATCH_SIZE = 30                            # Number of lines per API request

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def translate_chunk(chunk_lines, target_language):
    """
    Translates a chunk of lines using OpenAI's API. The prompt instructs the model to:
      - Translate the given English lines to the target language.
      - Return a JSON array where each element is an object containing:
          - "index": the line number in the chunk,
          - "english_content": the original line,
          - "translated_content": the translated line.
    """
    # Build the prompt
    prompt = (
        f"Translate the following {len(chunk_lines)} lines from English to {target_language}.\n"
        "For each line, output a JSON object with the following keys:\n"
        " - \"index\": the line's index (starting from 0),\n"
        " - \"english_content\": the original English line,\n"
        " - \"translated_content\": the translated text.\n"
        "Return only a valid JSON array of these objects without any extra text.\n\n"
    )
    # Append each line with its index information (optional for clarity, the index will be added in JSON)
    for idx, line in enumerate(chunk_lines):
        # If a line is empty, it should be preserved in the translation.
        prompt += f"{idx}: {line}\n"
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # or another model of your choice
            messages=[
                {"role": "system", "content": "You are a translation assistant for translating dialogues from spaceflight."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        translated_json = response.choices[0].message.content.strip()
        
        lines = translated_json.splitlines()
        translated_json = "\n".join(lines[1:-1])
        # Attempt to parse the JSON output.
        translations = json.loads(translated_json)
        # Verify that we have a list and the expected number of entries.
        if isinstance(translations, list) and len(translations) == len(chunk_lines):
            # Sort translations by index to ensure correct order
            translations.sort(key=lambda x: x.get("index", 0))
            # Extract only the translated text from each object
            translated_lines = [item.get("translated_content", "") for item in translations]
            return translated_lines
        else:
            print("Warning: JSON output format did not match expectations. Falling back to line-by-line translation.")
            return chunk_lines
    except Exception as e:
        print(f"Error during translation: {e}")
        return chunk_lines

def process_file(file_path, target_language):
    """
    Reads a file line by line, translates in batches of BATCH_SIZE using JSON output format,
    and writes the translated results to a new file.
    """
    print(f"Processing file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as infile:
        all_lines = infile.readlines()
    
    # Remove newline characters (we'll add them back when writing)
    all_lines = [line.rstrip("\n") for line in all_lines]
    translated_lines = []
    
    for i in tqdm(range(0, len(all_lines), BATCH_SIZE)):
        chunk = all_lines[i:i+BATCH_SIZE]
        translations = translate_chunk(chunk, target_language)
        translated_lines.extend(translations)
        # Sleep to manage rate limits; adjust or remove as needed.
        time.sleep(0.5)
    
    filename = os.path.basename(file_path)
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as outfile:
        for line in translated_lines:
            outfile.write(line + "\n")
    print(f"Saved translated file to: {output_path}")

def main():
    file_list = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    print(f"Found {len(file_list)} .txt files to process.")
    
    for file_path in file_list:
        process_file(file_path, TARGET_LANGUAGE)

if __name__ == "__main__":
    main()