import streamlit as st
from data.Img2Latex import LatexFormulaDataset
from data.utils import load_checkpoint
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import CONFIG
from models.resnet50_transformer import ResNet50_Transformer
# from models.VIT_Transformers import ViT_Transformer


hide_footer_style = """
    <style>
    footer {visibility: hidden;}
    </style>
# """
st.set_page_config(layout='wide')
st.markdown(hide_footer_style, unsafe_allow_html=True)

st.title('Image to Latex')
st.markdown('----')

test_transforms = A.Compose(
    [
        A.Normalize((0.5,), (0.5,), max_pixel_value=255),
        ToTensorV2(),
    ],
)


MODEL_PTH = r'model\img2latex_checkpoint.pth.tar'

val_csv = r'Datasets\Img2Latex\val.csv'
tokenizer_path = r'tokenizer.json'
base_dir = r'Datasets\Img2Latex'

special_characters = "{[(|,.&+-*/<=^%"


def convert_img2latex(image, model, vocabulary, search_algorithm='greedy', inference=False):
    augmented = test_transforms(image=image)
    image = augmented['image']

    image = torch.unsqueeze(image, 0).to(CONFIG.DEVICE)

    st.write(image.shape)
    if search_algorithm == 'beam':
        st.write("Using Beam Search")
        output = model.beam_search(image, 3)
    else:
        st.write("Using Greedy Search")
        output = model.greedy_search(image)

    flattened_output = output.flatten().tolist()

    output_token = ""

    finished = False
    for i in range(len(flattened_output)):
        token = vocabulary.index_to_token[str(flattened_output[i])]
        if token == '\\end':
            finished = True

        if not inference:
            if i >= 10 and not finished:
                if not token.startswith("\\") and token.isalpha():
                    previous_token = vocabulary.index_to_token[str(flattened_output[i - 1])]

                    # Check if the previous token contains '{'
                    if (not any(c in special_characters for c in previous_token) or
                            (len(previous_token) > 1 and previous_token != '\\begin' and previous_token != '\\end')):
                        token = "{" + token + "}"  # Add "\\" to the token

        output_token = output_token + token

        if flattened_output[i] == 2:
            break

    st.subheader("Model's numerical output: ")
    st.write(output)

    st.markdown('----')

    return output_token


def clean_and_write_latex(latex_str):
    # Clean the input by removing special tokens
    latex_str = latex_str.replace("<SOS>", "").replace("<EOS>", "").strip()
    latex_str = latex_str.replace("\\begin{align*}", "").replace("\\end{align*}", "").strip()
    st.write(f'**Cleaned Output:** {latex_str}')
    st.markdown('----')
    st.subheader("Rendered Formula:")

    st.latex(r"\Large " + latex_str)


val_dataset = LatexFormulaDataset(val_csv, base_dir, tokenizer_path, augmentations=test_transforms, train=False)
vocab = val_dataset.vocab

model = ResNet50_Transformer(
    d_model=512,
    dim_feedforward=1024,
    nhead=8,
    dropout=0.2,
    num_decoder_layers=5,
    max_output_len=300,
    pad_index=0,
    sos_index=1,
    eos_index=2,
    num_classes=len(val_dataset.vocab.token_to_index),
)


# model = ViT_Transformer(
#         d_model=CONFIG.d_model,
#         img_size=(96, 416),
#         dim_feedforward=CONFIG.dim_forward,
#         vit_emb=768,
#         nhead=CONFIG.n_head,
#         dropout=CONFIG.dropout,
#         num_encoder_layers=CONFIG.num_encoder_layers,
#         num_decoder_layers=CONFIG.num_decoder_layers,
#         max_output_len=CONFIG.max_output_len,
#         pad_index=0,
#         sos_index=1,
#         eos_index=2,
#         num_classes=len(val_dataset.vocab.token_to_index),
#         training=False
#     )

def set_bn_train(m):
    if isinstance(m, nn.BatchNorm2d):
        m.train()  # Keep BatchNorm in training mode


model = model.to(CONFIG.DEVICE)

# model.eval()  # Still keeps dropout disabled, only affects BN layers
# model.apply(set_bn_train)

model.eval()


# Ignore padding
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=CONFIG.milestones, gamma=CONFIG.gamma)

load_checkpoint(MODEL_PTH, model, optimizer, scheduler, scaler, CONFIG.lr)
st.write("==> Loaded model successfully!")

IMG_FILE = st.file_uploader("Upload Image file here")

if IMG_FILE:
    image = np.array(Image.open(IMG_FILE).convert("RGB"))

    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.axis('off')
    st.pyplot(fig)

    st.markdown('----')
    output_formula = convert_img2latex(image, model, vocab, 'greedy')
    st.write(f'**Raw Output:** {output_formula}')
    try:
        clean_and_write_latex(output_formula)

    except:
        st.write("*Error converting formula to image!*")
