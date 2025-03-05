import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5

SAVE_MODEL = True
LOAD_MODEL = True

# Data
batch_size = 32
num_workers = 4
pin_memory = True

# Optim
lr = 0.00002
weight_decay = 0.0001

# Scheduler
milestones = [5, 10, 15]
gamma = 0.5

# Model
d_model = 768
dim_forward = 1024
n_head = 8
dropout = 0.2
num_encoder_layers = 6
num_decoder_layers = 5
max_output_len = 300

MODEL_PATH = r'C:\Users\Daryn Bang\Desktop\Models\Image2Latex_leVIT\img2latex_checkpoint.pth.tar'
CHECKPOINT_PATH = r'C:\Users\Daryn Bang\Desktop\Models\Image2Latex_leVIT\img2latex_checkpoint_new.pth.tar'
