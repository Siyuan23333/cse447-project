from typing import List, Dict, Any

import torch

class TrainDataCollator:
    
    def __init__(
        self,
        pad_token_id: int,
        sentinel_token_id: int,
        decoder_start_token_id: int,
        fixed_encoder_length: int,
        fixed_mask_length: int,
        token_per_mask: int,
    ):
        self.pad_token_id = pad_token_id
        self.sentinel_token_id = sentinel_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.fixed_encoder_length = fixed_encoder_length
        self.fixed_mask_length = fixed_mask_length
        self.token_per_mask = token_per_mask

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        batch_decoder_input_ids = []
        batch_decoder_attention_masks = []
        
        for example in examples:
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            true_length = torch.sum(attention_mask, dim=-1) - 1
            
            mask_start_positions = self.create_mask_starting_positions(true_length)
            
            for mask_start in mask_start_positions:
                variant_input_ids, variant_attention_mask, variant_labels = self.create_variant(
                    input_ids, attention_mask, true_length, mask_start,
                )
                
                variant_decoder_input_ids = self.shift_tokens_right(variant_labels)
                variant_decoder_attention_mask = [
                    1 if token != self.pad_token_id else 0 for token in variant_decoder_input_ids
                ]
                
                batch_input_ids.append(variant_input_ids)
                batch_attention_masks.append(variant_attention_mask)
                batch_labels.append(variant_labels)
                batch_decoder_input_ids.append(variant_decoder_input_ids)
                batch_decoder_attention_masks.append(variant_decoder_attention_mask)
        
        # Convert lists to numpy arrays
        batch = {
            "input_ids": np.array(batch_input_ids, dtype=np.int32),
            "attention_mask": np.array(batch_attention_masks, dtype=np.int32),
            "labels": np.array(batch_labels, dtype=np.int32),
            "decoder_input_ids": np.array(batch_decoder_input_ids, dtype=np.int32),
            "decoder_attention_mask": np.array(batch_decoder_attention_masks, dtype=np.int32),
        }
        return batch
      
    def create_mask_starting_positions(self, true_length: int) -> List[int]:
        """
        Create starting positions for the masked tokens.
        
        The function generates a list of starting positions for the masked spans,
        
        ensuring that the spans do not exceed the true length of the input.
        """
        mask_start_positions = []
        num_masks = true_length // self.token_per_mask + 1
        for _ in range(num_masks):
            start_pos = np.random.randint(0, true_length)
            mask_start_positions.append(start_pos)
        
        return mask_start_positions

    def create_variant(self, input_ids: List[int], attention_mask: List[int], true_length: int, mask_start: int):
        """
        Create a variant of a single example by masking the last `mask_length` tokens.
        
        The encoder input keeps the tokens from the beginning up to (true_length - mask_length),
        then replaces the masked tokens with a single sentinel token, and pads to a fixed length.
        
        The labels are constructed as the sentinel token followed by the masked tokens,
        then padded to a fixed label length.
        """
        # Determine the split: unmasked tokens and masked tokens.
        unmasked_length = mask_start
        masked_tokens = input_ids[mask_start:true_length]
        
        # Build the new encoder input:
        # - Keep tokens [0:unmasked_length]
        # - Insert one sentinel token to represent the masked span
        encoder_tokens = input_ids[:unmasked_length] + [self.sentinel_token_id]
        # Pad encoder_tokens to fixed_encoder_length
        encoder_padding_length = self.fixed_encoder_length - len(encoder_tokens)
        encoder_tokens = encoder_tokens + [self.pad_token_id] * encoder_padding_length
        
        # Build the corresponding encoder attention mask:
        # 1 for the actual tokens (unmasked + sentinel), 0 for the padded tokens.
        encoder_attention = [1] * (unmasked_length + 1) + [0] * encoder_padding_length
        
        # Build the labels: start with the sentinel token followed by the masked tokens.
        labels = [self.sentinel_token_id] + masked_tokens
        labels_padding_length = self.fixed_label_length - len(labels)
        labels = labels + [self.pad_token_id] * labels_padding_length
        
        return encoder_tokens, encoder_attention, labels

    def shift_tokens_right(self, labels: List[int]) -> List[int]:
        """
        Shift labels to the right to produce decoder input ids.
        Insert the decoder_start_token_id at the beginning and remove the last token.
        """
        # Typically, shifting involves removing the last token and prepending the start token.
        
        return [self.decoder_start_token_id] + labels[:-1]


# === Example usage ===

# Suppose we have the following configuration:
PAD_TOKEN_ID = 0
SENTINEL_TOKEN_ID = 258   # Example: T5 uses extra_id tokens at the high end of the vocab.
DECODER_START_TOKEN_ID = PAD_TOKEN_ID  # For T5, this is typically 0 (or could be the pad token)
FIXED_ENCODER_LENGTH = 128   # For illustration (in practice, use the desired length)
FIXED_MASK_LENGTH = 16      # For illustration
TOKEN_PER_MASK = 16 

collator = TrainDataCollator(
    pad_token_id=PAD_TOKEN_ID,
    sentinel_token_id=SENTINEL_TOKEN_ID,
    decoder_start_token_id=DECODER_START_TOKEN_ID,
    fixed_encoder_length=FIXED_ENCODER_LENGTH,
    fixed_mask_length=FIXED_MASK_LENGTH,
    token_per_mask=TOKEN_PER_MASK,
)

# An example input: a single sentence that has been tokenized and padded.
example_sentence = {
    "input_ids": [101, 102, 103, 104, 105, 106, 107, 108, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
    # Assume the real sentence length is 8 tokens.
    "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1] + [0] * 8,
}

# Collate a batch (even if batch size is 1, the collator will produce multiple variants)
batch = collator([example_sentence])

# The output batch will have keys: "input_ids", "attention_mask", "labels", "decoder_input_ids", and "decoder_attention_mask".
print("input_ids:\n", batch["input_ids"])
print("attention_mask:\n", batch["attention_mask"])
print("labels:\n", batch["labels"])
print("decoder_input_ids:\n", batch["decoder_input_ids"])
print("decoder_attention_mask:\n", batch["decoder_attention_mask"])
