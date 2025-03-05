from Img2Latex.models.VIT_Transformers import ViT_Transformer
from data.utils import *
from data.Img2Latex import (LatexFormulaDataset,
                            get_dataloader,
                            train_transforms,
                            test_transforms)
import Img2Latex.CONFIG as CONFIG
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


tokenizer_path = r'Datasets\Img2Latex\tokenizer.json'

train_csv = r'Datasets\Img2Latex_3\train.csv'
val_csv = r'Datasets\Img2Latex_3\val.csv'
base_dir = r'Datasets\Img2Latex_3'

torch.backends.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scheduler):
    model.train()
    loop = tqdm(train_loader, leave=True)
    running_loss = 0.0

    for batch_idx, (imgs, targets) in enumerate(loop):
        x = imgs.to(CONFIG.DEVICE)
        y = targets.to(CONFIG.DEVICE)

        with torch.cuda.amp.autocast():
            logits = model(x, y[:, :-1])      # Teacher-forcing

            loss = loss_fn(logits, y[:, 1:])

            # for param in model.parameters():
            #     param.grad = None

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

    scheduler.step()

def lr_lambda(current_step, warmup_steps, base_lr, total_steps):
    if current_step < warmup_steps:
        return current_step / warmup_steps  # Linear warm-up

    min_lr_scale = 0.5  # Instead of decaying to 0, decay to 50% of base LR
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)

    return min_lr_scale + (1 - min_lr_scale) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate_fn(val_loader, model, loss_fn, cer_metric):
    model.eval()  # Set the model to evaluation mode
    loop = tqdm(val_loader, leave=False)
    running_loss = 0.0
    print("Evaluating Model.. ")
    cer_metric.reset()

    for batch_idx, (imgs, targets) in enumerate(loop):
        x = imgs.to(CONFIG.DEVICE, non_blocking=True)
        y = targets.to(CONFIG.DEVICE)

        with torch.amp.autocast(CONFIG.DEVICE):
            logits = model(x, y[:, :-1])
            loss = loss_fn(logits, y[:, 1:])

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN detected in loss at batch {batch_idx}")
                continue

        running_loss += loss.item()

        # Decode predictions and update CER
        preds = logits.argmax(dim=-1)
        cer_metric.update(preds, y)

    print(f"Running loss: {running_loss}")
    val_loss = running_loss / len(val_loader)
    val_cer = cer_metric.compute()
    return val_loss, val_cer

def main():
    print(CONFIG.DEVICE)
    print(CONFIG.lr)

    train_dataset = LatexFormulaDataset(train_csv, base_dir, tokenizer_path, augmentations=train_transforms, train=True)
    val_dataset = LatexFormulaDataset(val_csv, base_dir, tokenizer_path, augmentations=test_transforms, train=False)

    train_loader = get_dataloader(train_dataset)
    val_loader = get_dataloader(val_dataset)

    print(len(val_dataset.vocab.token_to_index))

    # 28 minutes
    # 48 minutes if mlp_dim is 1024, d_model is 768
    model = ViT_Transformer(
        d_model=CONFIG.d_model,
        img_size=(96, 416),
        dim_feedforward=CONFIG.dim_forward,
        vit_emb=768,
        nhead=CONFIG.n_head,
        dropout=CONFIG.dropout,
        num_encoder_layers=CONFIG.num_encoder_layers,
        num_decoder_layers=CONFIG.num_decoder_layers,
        max_output_len=CONFIG.max_output_len,
        pad_index=0,
        sos_index=1,
        eos_index=2,
        num_classes=len(val_dataset.vocab.token_to_index),
    )

    model.to(CONFIG.DEVICE)

    # Ignore padding
    loss_fn = nn.CrossEntropyLoss(ignore_index=0).to(CONFIG.DEVICE)
    val_cer = CharacterErrorRate(val_dataset.vocab.ignore_indices)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    scaler = torch.amp.GradScaler(CONFIG.DEVICE)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=CONFIG.milestones, gamma=CONFIG.gamma)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda step: lr_lambda(step, 3, CONFIG.lr, CONFIG.NUM_EPOCHS))

    if CONFIG.LOAD_MODEL:
        load_checkpoint(CONFIG.MODEL_PATH, model, optimizer, scheduler, scaler, CONFIG.lr)


    for epoch in range(CONFIG.NUM_EPOCHS):
        print(f"Current epoch: {epoch + 1}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scheduler)

        if CONFIG.SAVE_MODEL:
            save_checkpoint(model, optimizer, scheduler, scaler, filename=CONFIG.CHECKPOINT_PATH)

        if (epoch + 1) % 3 == 0:
            val_loss, val_cer = validate_fn(val_loader, model, loss_fn, val_cer)
            print(f"Validation Loss: {val_loss:.4f}, CER: {val_cer:.4f}")

        print("Finished training")


if __name__ == '__main__':
    main()
