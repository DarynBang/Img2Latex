import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .utils import Vocabulary
import numpy as np
from torch.utils.data import DataLoader
# from CNN_Architectures.Img2Latex import CONFIG
import CONFIG
from torch.nn.utils.rnn import pad_sequence
import os
from PIL import Image
import warnings
import pandas as pd

#
# from utils import Vocabulary
# import CONFIG

warnings.filterwarnings('ignore')

train_transforms = A.Compose(
    [
        A.Affine(scale=(0.6, 1.0), rotate=(-1, 1), shear=(-15, 15), p=0.5),
        A.GaussNoise(p=0.5),
        A.GaussianBlur(blur_limit=(1, 1), p=0.5),
        A.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
        A.Normalize((0.5,), (0.5,), max_pixel_value=255),
        ToTensorV2(),
    ],
)


test_transforms = A.Compose(
    [
        A.Normalize((0.5,), (0.5,), max_pixel_value=255),
        ToTensorV2(),
    ],
)

class LatexFormulaDataset(Dataset):
    def __init__(self, csv_path, base_dir, tokenizer_path, augmentations=None, train=True, freq_threshold=2):
        # initialise paramaters
        self.data = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.train = train
        self.augmentations = augmentations

        ## initialize vocabulary
        self.vocab = Vocabulary(freq_threshold)

        # Use existing vocabulary
        self.vocab.load_vocabulary(tokenizer_path)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_file = row["image_path"]
        formula = row["latex_formula"]

        # Load and preprocess the image
        image_path = os.path.join(self.base_dir, image_file)
        try:
            image = np.array(Image.open(image_path).convert("RGB"))  # Ensure RGB format
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        # Tokenize formula
        num_tex = [self.vocab.token_to_index["<SOS>"]]
        num_tex += self.vocab.numericalize(formula)
        num_tex.append(self.vocab.token_to_index["<EOS>"])

        return image, torch.tensor(num_tex, dtype=torch.long)


def collate_function(batch):
    """
        Args:
            batch: List of tuples (image, tokenized_formula_tensor)

        Returns:
            images
            padded_formulas: Padded formula tensors of shape (B, max_seq_len)
        """

    images, formulas = zip(*batch)

    images = torch.stack(images)

    # Get formula max length
    max_length = max(len(formula) for formula in formulas)

    # Initialize tensors for padded images and formulas
    B = len(images)
    padded_formulas = torch.full((B, max_length + 2), fill_value=0, dtype=torch.long)

    # Populate the tensors
    for i, (image, formula) in enumerate(batch):
        padded_formulas[i, :formula.size(0)] = formula  # Copy formula into the padded tensor

    return images, padded_formulas


def get_dataloader(dataset, batch_size=CONFIG.batch_size):
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=CONFIG.num_workers,
        pin_memory=CONFIG.pin_memory,
        collate_fn=collate_function,
    )


def filter_by_width(example):
    """
    Filter function to keep only examples where the image width
    is within a certain range.
    """
    width_threshold_min = 50  # Set your minimum width threshold
    width_threshold_max = 500  # Set your maximum width threshold

    image = example["image"]
    width = image.size[0]  # Assuming the image is in PIL format
    return width_threshold_min <= width <= width_threshold_max


def main():
    dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")
    filtered_dataset = dataset.filter(filter_by_width)
    train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_ds = train_val_split['train']    # Training dataset
    val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)
    val_ds = val_test_split["train"]    # Validation dataset
    test_ds = val_test_split["test"]    # Test dataset

    tokenizer_path = CONFIG.tokenizer_json

    val_dataset = LatexFormulaDataset(val_ds, tokenizer_path, augmentations=test_transforms)
    print(len(val_dataset.vocab.token_to_index))
    print(val_dataset[1])


    val_loader = get_dataloader(val_dataset)

    # Example: Iterate through DataLoader
    for batch in val_loader:
        images, formulas = batch
        print(images.shape)  # (B, C, max_H, max_W)
        print(formulas.shape)  # (B, max_seq_len)


if __name__ == '__main__':
    main()
