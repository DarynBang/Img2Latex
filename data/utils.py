import re
from datasets import load_dataset
import cv2
import torch
import torch.nn as nn
from typing import Set
import editdistance
import albumentations as A
from torch import Tensor
from torchmetrics import Metric
import math
import numpy as np


OUTPUT_DIR = r"Datasets\Im2Latex\tokenizer.json"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, scheduler, scaler, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, scheduler, scaler, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location='cuda')

    # Load states
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    # Update learning rate to the desired value
    print("Updating Learning Rate")
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Vocabulary:
    def __init__(self, min_count: int = 2):
        """
        :param min_count: words appearing less than min_count will be excluded
        """
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.index_to_token = {i: tok for i, tok in enumerate(self.special_tokens)}
        self.token_to_index = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.ignore_indices = set(range(len(self.special_tokens)))

        self.min_count = min_count

        # self.TOKENIZE_PATTERN = re.compile(
        #     "(\\\\[a-zA-Z]+)|" +
        #     '((\\\\)*[$-/:-?{-~!"^_`\[\]])|' +
        #     "(\w)|" +
        #     "(\\\\)"
        # )

        self.TOKENIZE_PATTERN = re.compile(
            "(\\\\[a-zA-Z]+)|" + '((?<!\\\\)[$-/:-?{-~!"^_`\[\]])|' + "(\w)"
        )

    def __len__(self):
        return len(self.index_to_token)

    @staticmethod
    def tokenizer(TOKENIZE_PATTERN, formula: str, inference: bool = False):
        tokens = re.finditer(TOKENIZE_PATTERN, formula)
        tokens = list(map(lambda x: x.group(0), tokens))
        if not inference:
            tokens = [x for x in tokens if x is not None and x != ""]

        else:
            tokens = [x for x in tokens if x is not None and x != ""]
            tokens.insert(0, "<SOS>")
            tokens.append("<EOS>")


        return tokens

    def build_vocabulary(self, sentence_list):
        count = {}

        index = 4

        for sentence in sentence_list:
            for word in self.tokenizer(self.TOKENIZE_PATTERN, sentence):
                if word not in count:
                    count[word] = 1
                else:
                    count[word] += 1

                if word not in self.token_to_index and count[word] > self.min_count - 1:
                    self.index_to_token[index] = word
                    self.token_to_index[word] = index

                    index += 1

    def numericalize(self, sentence, inference=False):
        tokenized_text = self.tokenizer(self.TOKENIZE_PATTERN, sentence, inference=inference)

        return [
            self.token_to_index.get(word, self.token_to_index["<UNK>"])
            for word in tokenized_text
        ]


    def save_vocabulary(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'token_to_index': self.token_to_index,
                'index_to_token': self.index_to_token
            }, f)

    def load_vocabulary(self, filepath):
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.token_to_index = data['token_to_index']
            self.index_to_token = data['index_to_token']


class CharacterErrorRate(Metric):
    def __init__(self, ignore_indices: Set[int], *args):
        super().__init__(*args)
        self.ignore_indices = ignore_indices
        self.add_state('error', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.error: Tensor
        self.total: Tensor

    def reset(self):
        """Reset the CER metric to its initial state at the beginning of each epoch."""
        self.error.fill_(0)
        self.total.fill_(0)

    def update(self, preds, targets):
        N = preds.shape[0]
        for i in range(N):
            # print(f'Predictions: {preds}')
            # print(f'Targets: {target}')
            pred = [token for token in preds[i].tolist() if token not in self.ignore_indices]
            target = [token for token in targets[i].tolist() if token not in self.ignore_indices]

            # Computes Levenshtein distance
            distance = editdistance.distance(pred, target)

            if max(len(pred), len(target)) > 0:
                self.error += distance / max(len(pred), len(target))
        self.total += N

    def compute(self):
        return self.error / self.total


def filter_by_height(example):
    """
    Filter function to keep only examples where the image width
    is within a certain range.
    """
    height_threshold_max = 300  # Set your maximum width threshold

    image = example["image"]
    height = image.size[1]  # Assuming the image is in PIL format
    return height <= height_threshold_max


def filter_by_length(example):
    """
    Filter function to keep only examples where the image width
    is within a certain range.
    """
    length_threshold_max = 375  # Set your maximum width threshold

    seq = example["latex_formula"]
    length = len(seq)  # Assuming the image is in PIL format
    return length <= length_threshold_max

def resize(image):
    IMG_HEIGHT, IMG_WIDTH = 96, 384
    transform = A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH, interpolation=cv2.INTER_LANCZOS4),
    ])

    transformed = transform(image=np.array(image))
    image_arr = transformed['image']

    return image_arr


def main():
    # dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")
    # print(len(dataset['train']))
    #
    # train_val_split = dataset['train'].train_test_split(test_size=0.6, seed=42)
    # train_ds = train_val_split['train']

    v = Vocabulary(min_count=2)
    v.load_vocabulary(r'Datasets\Img2Latex\tokenizer.json')
    print("Loaded vocabulary successfully! ")

    latex = r'\begin{align*} L_{\vec{X}} \phi (\vec{X}) = \mbox{Tr}[J] \phi (\vec{X}) + P (\vec{V} \cdot \vec{\gamma})\end{align*}'

    print(v.index_to_token)
    print(v.numericalize(latex))



if __name__ == '__main__':
    main()

"""
Max width: 746 at index 28433
Max height: 386 at index 305348
"""
