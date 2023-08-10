import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Any

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        # Initialize the BilingualDataset with the provided parameters.
        super().__init__()
        # ds: The original dataset containing bilingual sentences.
        self.ds = ds
        # tokenizer_src: The tokenizer for the source language.
        self.tokenizer_src = tokenizer_src
        # tokenizer_tgt: The tokenizer for the target language.
        self.tokenizer_tgt = tokenizer_tgt
        # src_lang: The source language code.
        self.src_lang = src_lang
        # tgt_lang: The target language code.
        self.tgt_lang = tgt_lang
        # seq_len: The maximum sequence length to which sentences will be truncated or padded.
        self.seq_len = seq_len

        # Create tensors for special tokens to be used later during processing.
        # Create a tensor for the start-of-sentence token for the source language.
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        # Create a tensor for the end-of-sentence token for the source language.
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        # Create a tensor for the padding token for the source language.
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        # Return the length of the dataset.
        return len(self.ds)


    def __getitem__(self, idx):
        # Get the source and target text pair for the given index
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into tokens and then into IDs
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add SOS, EOS, and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for SOS and EOS tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # -1 for EOS token

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Create tensors for the encoder input, decoder input, and the label
        # Add SOS and EOS to the encoder input and pad it
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add SOS to the decoder input and pad it
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Ensure that the sizes of the tensors are correct
        assert encoder_input.size(0) == self.seq_len, "Encoder input is not of the right size"
        assert decoder_input.size(0) == self.seq_len, "Decoder input is not of the right size"
        assert label.size(0) == self.seq_len, "Label is not of the right size"

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(seq_len):
    # Create a mask that prevents the decoder from attending to future positions.
    # The mask will have 0s above the main diagonal and 1s below it.
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int)
    return mask == 0