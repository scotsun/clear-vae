import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
import re

PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

tokenizer = get_tokenizer("spacy", language="en_core_web_sm")


def clean_text(text: str):
    text = text.lower().strip()
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z!?]+", r" ", text)
    return text


def yield_tokens(dataset, tokenizer=tokenizer):
    for _, _row in tqdm(dataset.iterrows()):
        yield tokenizer(clean_text(_row["review"]))


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids, type: str):
    match type:
        case "eos":
            return torch.cat([torch.tensor(token_ids), torch.tensor([EOS_IDX])])
        case "bos":
            return torch.cat([torch.tensor([BOS_IDX]), torch.tensor(token_ids)])
        case _:
            raise ValueError("incorrect value for type.")


class Collator:
    def __init__(self, vocab) -> None:
        self.vocab = vocab

    def __call__(self, batch) -> tuple:
        eos_transform = sequential_transforms(
            tokenizer, self.vocab, lambda x: tensor_transform(x, "eos")
        )

        label_batch, eos_batch = [], []
        for label, text in batch:
            label_batch.append(label)
            eos_batch.append(eos_transform(clean_text(text)))

        eos_batch = pad_sequence(eos_batch, padding_value=PAD_IDX, batch_first=True)

        return label_batch, eos_batch
