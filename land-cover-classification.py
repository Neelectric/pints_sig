import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datasets import load_from_disk, load_dataset
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report
import wandb
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add the missing SubsetWithTransform class
class SubsetWithTransform(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
        # These attributes will be copied from the parent dataset later

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        if self.transform:
            item['image'] = self.transform(item['image'])
        return item

# Initialize wandb for experiment tracking (optional but recommended)
wandb.init(project="uk-landcover-classification", name="unet-training")

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Configuration
class Config:
    img_size = 336  # Image size for model input
    batch_size = 2  # Batch size for training
    num_workers = 0 # Number of workers for data loading
    num_epochs = 3  # Number of training epochs
    learning_rate = 1e-4  # Learning rate
    weight_decay = 0.05  # Weight decay for regularization
    warmup_steps = 100  # Warmup steps for learning rate scheduler
    mixed_precision = True  # Use mixed precision training
    gradient_accumulation_steps = 1  # Gradient accumulation steps
    prefetch_factor = None  # Prefetch factor for data loading
    pin_memory = True  # Pin memory for faster data transfer
    persistent_workers = None  # Keep workers alive between epochs
    
config = Config()
dataset = load_dataset("pints-sig/uk_landcover")

ds = dataset["train"].train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]

from transformers import AutoImageProcessor

checkpoint = "nvidia/mit-b0"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)

def train_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["label"]]
    inputs = image_processor(images, labels)
    return inputs

train_ds.set_transform(train_transforms)
test_ds.set_transform(train_transforms)

id2label = {
    1: "Deciduous woodland",
    2: "Coniferous woodland",
    3: "Arable",
    4: "Improve grassland",
    5: "Neutral grassland",
    6: "Calcareous grassland",
    7: "Acid grassland",
    8: "Fen",
    9: "Heather",
    10: "Heather grassland",
    11: "Bog",
    12: "Inland rock",
    13: "Saltwater",
    14: "Freshwater",
    15: "Supralittoral rock",
    16: "Supralittoral sediment",
    17: "Littoral rock",
    18: "Littoral sediment",
    19: "Saltmarsh",
    20: "Urban",
    21: "Suburban"
}

label2id = {
    "Deciduous woodland": 1,
    "Coniferous woodland": 2,
    "Arable": 3,
    "Improve grassland": 4,
    "Neutral grassland": 5,
    "Calcareous grassland": 6,
    "Acid grassland": 7,
    "Fen": 8,
    "Heather": 9,
    "Heather grassland": 10,
    "Bog": 11,
    "Inland rock": 12,
    "Saltwater": 13,
    "Freshwater": 14,
    "Supralittoral rock": 15,
    "Supralittoral sediment": 16,
    "Littoral rock": 17,
    "Littoral sediment": 18,
    "Saltmarsh": 19,
    "Urban": 20,
    "Suburban": 21
}


from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)



# Calculate total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Initialize model
model = model.cuda()



training_args = TrainingArguments(
    output_dir="segformer-b0-scene-parse-150",
    learning_rate=6e-5,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_total_limit=3,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    remove_unused_columns=False,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()