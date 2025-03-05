from datasets import load_dataset
from PIL import Image, ImageOps
import os
import pandas as pd
from tqdm import tqdm

IMG_HEIGHT, IMG_WIDTH = 96, 384

MIN_WIDTH = 60
MAX_WIDTH = 550
MAX_HEIGHT = 150

def filter_by_width(example):
    """
    Filter function to keep only examples where the image width
    is within a certain range.
    """
    width_threshold_min = MIN_WIDTH  # Set your maximum width threshold
    width_threshold_max = MAX_WIDTH  # Set your maximum width threshold

    image = example["image"]
    width = image.size[0]  # Assuming the image is in PIL format
    return width_threshold_min <= width <= width_threshold_max


def filter_by_height(example):
    """
    Filter function to keep only examples where the image height
    is within a certain range.
    """
    height_threshold_max = MAX_HEIGHT  # Set your maximum width threshold

    image = example["image"]
    height = image.size[1]  # Assuming the image is in PIL format
    return height <= height_threshold_max


def filter_by_length(example):
    """
    Filter function to keep only examples where the image width
    is within a certain range.
    """
    length_threshold_max = 300  # Set your maximum width threshold

    seq = example["latex_formula"]
    length = len(seq)  # Assuming the image is in PIL format
    return length <= length_threshold_max


# Directory to save resized images
base_dir = r'Datasets\Img2Latex'
os.makedirs(base_dir, exist_ok=True)

def resize_and_save(idx, instance, output_dir, size=(IMG_WIDTH, IMG_HEIGHT), WIDTH_THRESH=128, HEIGHT_THRESH=32):
    os.makedirs(output_dir, exist_ok=True)
    image = instance["image"]
    image_w, image_h = image.size

    # Pad image if smaller than minimum dimensions
    pad_width = max(WIDTH_THRESH - image_w, 0)
    pad_height = max(HEIGHT_THRESH - image_h, 0)

    if pad_width > 0 or pad_height > 0:
        padding = (
            pad_width // 2,  # Left
            pad_height // 2,  # Top
            pad_width - pad_width // 2,  # Right
            pad_height - pad_height // 2  # Bottom
        )
        image = ImageOps.expand(image, padding, fill='white')

    # Resize image
    image_resized = image.resize(size, Image.LANCZOS)

    # Save resized image
    output_path = os.path.join(output_dir, f"im2latex_{idx}.png")
    image_resized.save(output_path)
    return output_path

# Load the dataset
dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")


# Replace with actual conditions
filtered_dataset = dataset.filter(filter_by_length)
filtered_dataset = filtered_dataset.filter(filter_by_height)
filtered_dataset = filtered_dataset.filter(filter_by_width)

# Split dataset
train_val_split = filtered_dataset["train"].train_test_split(train_size=0.75, seed=32)   # train 0.75 and remaining 0.25
train_ds = train_val_split["train"]
val_test_split = train_val_split["test"].train_test_split(train_size=0.4, seed=32)     # val 0.05 and test 0.2
val_ds = val_test_split["train"]
test_ds = val_test_split["test"]


# Save function
def save(dataset, split_type):
    print(f"Saving for {split_type} data")
    split_dir = os.path.join(base_dir, split_type)
    data = []
    for idx, instance in tqdm(enumerate(dataset)):  # Iterate over the dataset split
        resize_and_save(idx, instance, split_dir)

        img_path = os.path.join(split_type, f'im2latex_{idx}.png').replace('\\', '/')
        data.append({"image_path": img_path, "latex_formula": instance["latex_formula"]})

    # Save CSV
    csv_path = os.path.join(base_dir, f"{split_type}.csv")
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Resized images for {split_type} saved to {split_dir}.")
    print(f"CSV for {split_type} saved to {csv_path}.")


# Save train, val, and test splits
save(train_ds, 'train')
save(val_ds, 'val')
# save(test_ds, 'test')

