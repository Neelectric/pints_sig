import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms
import timm
from transformers import get_cosine_schedule_with_warmup
from transformers import ViTImageProcessor
from sklearn.metrics import classification_report
import wandb
from tqdm import tqdm
import numpy as np
from PIL import Image

# Initialize wandb for experiment tracking (optional but recommended)
# wandb.init(project="uk-landcover-classification", name="vit-large-training")

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Configuration
class Config:
    model_name = "vit_large_patch14_clip_336"  # Using ViT-Large with larger patch size and resolution
    img_size = 336  # Increased image size for better performance
    batch_size = 32  # Adjust based on memory constraints
    num_workers = 8
    num_epochs = 30
    learning_rate = 1e-4
    weight_decay = 0.05
    warmup_steps = 500
    mixed_precision = True  # Use mixed precision training for efficiency
    gradient_accumulation_steps = 2  # Accumulate gradients to simulate larger batch sizes
    
config = Config()


model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)
dataset = load_from_disk("./uk_landcover_dataset")
print(dataset)
print(len(dataset[0]))