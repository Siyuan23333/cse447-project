import unicodedata
def normalize_unicode(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def create_sequences(text, seq_length=100):
    sequences = []
    for i in range(len(text) - seq_length):
        sequences.append(text[i:i + seq_length + 1])  # Input + Next character
    return sequences

def create_char_mapping(text):
    chars = sorted(set(text))  # Get unique characters
    char_to_int = {char: i for i, char in enumerate(chars)}
    int_to_char = {i: char for i, char in enumerate(chars)}
    return char_to_int, int_to_char

def text_to_int(sequence, char_to_int):
    return [char_to_int[char] for char in sequence]
