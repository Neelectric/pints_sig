import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datasets import load_from_disk
from torchvision import transforms
import timm
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
wandb.init(project="uk-landcover-classification", name="vit-large-training")

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Configuration
class Config:
    model_name = "vit_large_patch14_clip_336"  # Using ViT-Large with larger patch size and resolution
    img_size = 336  # Increased image size for better performance
    batch_size = 4  # Smaller batch size to handle larger images
    num_workers = 8  # Using 32 processes as requested
    num_epochs = 3    # Using 3 epochs as requested
    learning_rate = 1e-4
    weight_decay = 0.05
    warmup_steps = 1000
    mixed_precision = True  # Use mixed precision training for efficiency
    gradient_accumulation_steps = 4  # Increased to compensate for smaller batch size
    prefetch_factor = 4  # Speed up data loading
    pin_memory = True  # Faster data transfer to GPU
    persistent_workers = True  # Keep workers alive between epochs
    
config = Config()

# Data Preparation
# More robust data transforms
def get_transforms(img_size):
    train_transform = transforms.Compose([
        # Convert various input types to PIL Image if needed
        transforms.Lambda(lambda x: x if isinstance(x, Image.Image) else Image.fromarray(np.array(x)) if isinstance(x, (list, np.ndarray)) else Image.new('RGB', (img_size, img_size), color=(100, 100, 100))),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Add a final check to ensure we have a valid tensor
        transforms.Lambda(lambda x: x if x.shape[0] == 3 else torch.zeros(3, img_size, img_size)),
    ])
    
    val_transform = transforms.Compose([
        # Convert various input types to PIL Image if needed
        transforms.Lambda(lambda x: x if isinstance(x, Image.Image) else Image.fromarray(np.array(x)) if isinstance(x, (list, np.ndarray)) else Image.new('RGB', (img_size, img_size), color=(100, 100, 100))),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Add a final check to ensure we have a valid tensor
        transforms.Lambda(lambda x: x if x.shape[0] == 3 else torch.zeros(3, img_size, img_size)),
    ])
    
    return train_transform, val_transform

# Load dataset with detailed error handling
try:
    print("Loading dataset from disk...")
    dataset = load_from_disk("./uk_landcover_dataset")
    print(f"Dataset loaded successfully. Type: {type(dataset)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# More efficient image conversion with shared cache
class UKLandcoverDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None, split='train'):
        self.dataset = hf_dataset[split] if split in hf_dataset else hf_dataset
        self.transform = transform
        self._cache = {}  # Simple cache for processed images
        self._cache_max_size = 1000  # Limit cache size to avoid memory issues
        
        # Function to extract label from potentially nested structures
        def get_label(item):
            label = item['label']
            # Handle if label is a list (could be nested)
            if isinstance(label, list):
                # If it's a list of lists, extract first element from first list
                if len(label) > 0:
                    if isinstance(label[0], list):
                        if len(label[0]) > 0:
                            return str(label[0][0])  # Convert to string for consistency
                        else:
                            return "unknown"
                    else:
                        return str(label[0])  # Convert to string for consistency
                else:
                    return "unknown"
            # If not a list, still convert to string for consistency
            return str(label)
        
        # Print a few samples to understand the structure
        print("Sample label structure examples:")
        for i in range(min(5, len(self.dataset))):
            print(f"Example {i}: {self.dataset[i]['label']} -> {get_label(self.dataset[i])}")
        
        # Get all unique labels as strings to ensure hashability
        sample_size = min(1000, len(self.dataset))
        sample_indices = np.random.choice(len(self.dataset), sample_size, replace=False)
        sample_labels = []
        
        # Extract labels and explicitly ensure they're hashable by converting to strings
        for i in sample_indices:
            try:
                extracted_label = get_label(self.dataset[i])
                sample_labels.append(extracted_label)
            except Exception as e:
                print(f"Error processing label at index {i}: {e}")
                sample_labels.append("error")
        
        # Create unique class list from string labels
        unique_labels = []
        unique_label_set = set()
        for label in sample_labels:
            if label not in unique_label_set:
                unique_label_set.add(label)
                unique_labels.append(label)
        
        self.labels = sorted(unique_labels)
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.num_classes = len(self.labels)
        
        # Store the get_label function for use in __getitem__
        self.get_label = get_label
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Sample of class labels: {self.labels[:5]} ...")
        print(f"Dataset size: {len(self.dataset)}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Important fix: convert numpy.int64 to regular int
        if isinstance(idx, np.int64):
            idx = int(idx)
            
        # Check if processed image is in cache
        if idx in self._cache:
            # Use cached data
            cached_data = self._cache[idx]
            if self.transform and 'image' in cached_data:
                # Apply transform to the cached image
                try:
                    cached_data['image'] = self.transform(cached_data['image'])
                except Exception as e:
                    if idx % 1000 == 0:
                        print(f"Transform error on cached image {idx}: {str(e)[:50]}")
                    cached_data['image'] = torch.zeros((3, 224, 224), dtype=torch.float32)
            return cached_data
        
        # If not in cache, process normally
        item = self.dataset[idx]
        result = {}
        
        # Handle image data with more robust type checking and caching
        try:
            image = item['image']
            
            # Convert various image formats to PIL Image
            if isinstance(image, str):  # If it's a path
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):  # If it's a numpy array
                image = Image.fromarray(image)
            elif isinstance(image, list):  # If it's a list
                # Handle list of pixel values - try different approaches
                try:
                    # Start with direct conversion to numpy
                    img_array = np.array(image)
                    
                    # Check dimensionality and handle appropriately
                    if len(img_array.shape) == 1:  # 1D array
                        # Try to determine if it's a flattened RGB image
                        if len(img_array) % 3 == 0:  # Multiple of 3 for RGB
                            size = int(np.sqrt(len(img_array) / 3))
                            try:
                                img_array = img_array.reshape(size, size, 3)
                            except:
                                # Alternative: try making it square
                                size = int(len(img_array) / 3)
                                img_array = img_array[:size*3].reshape(size, 1, 3)
                        else:
                            # Not a clean multiple of 3, might be grayscale
                            size = int(np.sqrt(len(img_array)))
                            try:
                                img_array = img_array.reshape(size, size, 1)
                                # Convert to 3 channels
                                img_array = np.repeat(img_array, 3, axis=2)
                            except:
                                # Last resort - create placeholder
                                if idx % 1000 == 0:
                                    print(f"Using placeholder for item {idx}")
                                img_array = np.ones((224, 224, 3), dtype=np.uint8) * 128
                    
                    # Ensure correct dtype
                    img_array = img_array.astype(np.uint8)
                    image = Image.fromarray(img_array)
                    
                except Exception as e:
                    if idx % 1000 == 0:
                        print(f"List conversion error at idx {idx}: {str(e)[:50]}")
                    image = Image.new('RGB', (224, 224), color=(100, 100, 100))
            else:
                # Unknown type - use placeholder
                if idx % 1000 == 0:
                    print(f"Unknown image type at idx {idx}: {type(image)}")
                image = Image.new('RGB', (224, 224), color=(100, 100, 100))
            
            # Store the PIL image in result
            result['image'] = image
            
        except Exception as e:
            if idx % 1000 == 0:
                print(f"Image processing error at idx {idx}: {str(e)[:50]}")
            # Fallback to placeholder
            result['image'] = Image.new('RGB', (224, 224), color=(100, 100, 100))
        
        # Get the label using our extraction function
        try:
            raw_label = self.get_label(item)
            
            # Convert label to numeric, handle unknown labels
            try:
                label = self.label2id[raw_label]
            except KeyError:
                # Only print warnings occasionally and not the full label 
                if idx % 1000 == 0:
                    truncated_label = str(raw_label)[:20] + "..." if len(str(raw_label)) > 20 else str(raw_label)
                    print(f"Unknown label at index {idx}: {truncated_label}")
                label = 0
            
            result['label'] = label
            
        except Exception as e:
            if idx % 1000 == 0:
                print(f"Label error at index {idx}: {str(e)[:50]}")
            result['label'] = 0
        
        # Get coordinates with error handling
        try:
            longitude = item.get('longitude', 0)
            latitude = item.get('latitude', 0)
            
            # Handle if coordinates are lists
            if isinstance(longitude, list) and len(longitude) > 0:
                longitude = longitude[0]
            if isinstance(latitude, list) and len(latitude) > 0:
                latitude = latitude[0]
            
            result['coords'] = torch.tensor([float(longitude), float(latitude)], dtype=torch.float)
        except Exception as e:
            if idx % 1000 == 0:
                print(f"Coordinate error at index {idx}: {str(e)[:50]}")
            result['coords'] = torch.tensor([0.0, 0.0], dtype=torch.float)
        
        # Add other fields
        result['filename'] = item.get('filename', '')
        result['is_valid'] = item.get('is_valid', True)
        
        # Cache the result (without transform applied)
        if len(self._cache) < self._cache_max_size:
            self._cache[idx] = result.copy()
        
        # Apply transform if needed
        if self.transform and 'image' in result:
            try:
                result['image'] = self.transform(result['image'])
            except Exception as e:
                if idx % 1000 == 0:
                    print(f"Transform error at idx {idx}: {str(e)[:50]}")
                result['image'] = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        return result

# Process dataset with UKLandcoverDataset wrapper to extract labels first
print("Creating initial dataset wrapper to determine classes...")
# Create a temporary wrapper to extract label information
temp_dataset = UKLandcoverDataset(dataset, transform=None)

# Get transforms
train_transform, val_transform = get_transforms(config.img_size)

# Now create the train/val split with the SubsetWithTransform class
print("Creating train/validation split...")
try:
    # Check for existing dataset splits
    if hasattr(dataset, 'keys'):
        dataset_keys = list(dataset.keys())
        print(f"Dataset contains the following splits: {dataset_keys}")
        
        if 'train' in dataset_keys and 'validation' in dataset_keys:
            print("Using existing train/validation split")
            train_dataset = dataset['train']
            val_dataset = dataset['validation']
        elif 'train' in dataset_keys and 'test' in dataset_keys:
            print("Using existing train/test split")
            train_dataset = dataset['train']
            val_dataset = dataset['test']
        else:
            raise ValueError("Dataset has keys but no train/val splits")
    else:
        has_train = hasattr(dataset, 'train')
        has_validation = hasattr(dataset, 'validation')
        has_test = hasattr(dataset, 'test')
        
        print(f"Dataset attribute check - train: {has_train}, validation: {has_validation}, test: {has_test}")
        
        if has_train and has_validation:
            print("Using train/validation attributes")
            train_dataset = dataset.train
            val_dataset = dataset.validation
        elif has_train and has_test:
            print("Using train/test attributes")
            train_dataset = dataset.train
            val_dataset = dataset.test
        else:
            # If no splits found, create a custom split
            raise ValueError("No valid splits found as attributes")
            
except Exception as e:
    print(f"Creating custom train/test split: {e}")
    
    # For large datasets, use a smaller validation set (10%)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Create indices for splitting
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create the subsets and copy attributes from the temp_dataset
    train_dataset = SubsetWithTransform(dataset, train_indices, transform=train_transform)
    val_dataset = SubsetWithTransform(dataset, val_indices, transform=val_transform)
    
    # Copy important attributes from the temp_dataset
    for ds in [train_dataset, val_dataset]:
        ds.labels = temp_dataset.labels
        ds.label2id = temp_dataset.label2id 
        ds.id2label = temp_dataset.id2label
        ds.num_classes = temp_dataset.num_classes
        ds.get_label = temp_dataset.get_label
    
    print(f"Created custom split: {len(train_dataset)} train samples, {len(val_dataset)} validation samples")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Sample class labels: {train_dataset.labels[:5]} ...")

# Create data loaders with reduced memory usage settings
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
    drop_last=True,
    prefetch_factor=config.prefetch_factor,
    persistent_workers=config.persistent_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
    prefetch_factor=config.prefetch_factor,
    persistent_workers=config.persistent_workers
)

# Model Definition - using the timm library for state-of-the-art vision models
class ViTForLandcover(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
        # Optional: incorporate geospatial coordinates
        # You could add these as additional features to the classification head
        self.use_coords = False
        if self.use_coords:
            # Get the classifier from the model
            in_features = self.model.head.in_features
            # Replace the classifier with one that uses image features + coordinates
            self.model.head = nn.Identity()
            self.coord_encoder = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, 256)
            )
            self.classifier = nn.Linear(in_features + 256, num_classes)
        
    def forward(self, images, coords=None):
        if self.use_coords and coords is not None:
            features = self.model(images)  # Extract features
            coord_features = self.coord_encoder(coords)  # Encode coordinates
            combined = torch.cat([features, coord_features], dim=1)
            return self.classifier(combined)
        else:
            return self.model(images)

# Configure model initialization with proper number of classes
def init_model(model_name, num_classes, pretrained=True):
    print(f"Initializing model with {num_classes} output classes")
    model = ViTForLandcover(model_name, num_classes, pretrained=pretrained)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    return model

# Initialize model with proper class count
model = init_model(config.model_name, train_dataset.num_classes)

# Configure for multiple GPUs
print(f"Detected {torch.cuda.device_count()} GPUs")
if torch.cuda.device_count() > 0:
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1:
        print(f"GPU 1: {torch.cuda.get_device_name(1)}")

# Use DataParallel if multiple GPUs are available
model = model.cuda()
if torch.cuda.device_count() > 1:
    print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
else:
    print("Using single GPU")

# Optimization
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

# Learning rate scheduler
num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.warmup_steps,
    num_training_steps=num_training_steps
)

# Loss function
criterion = nn.CrossEntropyLoss()

# Fix the deprecated GradScaler warning
if config.mixed_precision:
    print("Using mixed precision training")
    # Use the new recommended way to create a GradScaler
    try:
        scaler = torch.amp.GradScaler('cuda')
    except:
        # Fallback for older PyTorch versions
        print("Using legacy GradScaler API")
        scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

# Training function
def train_epoch(model, loader, optimizer, criterion, scheduler, scaler, epoch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}")
    
    # Count successful and failed batches
    successful_batches = 0
    failed_batches = 0
    
    for i, batch in enumerate(progress_bar):
        try:
            # Get data from batch with error handling
            try:
                images = batch['image'].cuda()
                labels = batch['label'].cuda()
                coords = batch.get('coords', None)
                if coords is not None:
                    coords = coords.cuda()
            except Exception as e:
                failed_batches += 1
                if i % 50 == 0:
                    print(f"Batch {i} error: {str(e)[:50]}")
                continue  # Skip this batch
            
            # Mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images, coords) if coords is not None else model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / config.gradient_accumulation_steps
                
                # Use scaler for gradient scaling
                scaler.scale(loss).backward()
                
                if ((i + 1) % config.gradient_accumulation_steps == 0) or (i + 1 == len(loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(images, coords) if coords is not None else model(images)
                loss = criterion(outputs, labels)
                loss = loss / config.gradient_accumulation_steps
                
                loss.backward()
                
                if ((i + 1) % config.gradient_accumulation_steps == 0) or (i + 1 == len(loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            running_loss += loss.item() * config.gradient_accumulation_steps
            progress_bar.set_postfix({
                "loss": f"{running_loss / (successful_batches + 1):.4f}",
                "ok/bad": f"{successful_batches}/{failed_batches}"
            })
            successful_batches += 1
            
        except Exception as e:
            failed_batches += 1
            if i % 50 == 0:
                print(f"Train error batch {i}: {str(e)[:50]}")
            continue
    
    epoch_loss = running_loss / successful_batches if successful_batches > 0 else float('inf')
    print(f"Epoch {epoch+1}: {successful_batches} successful, {failed_batches} failed batches. Loss: {epoch_loss:.4f}")
    return epoch_loss

# Validation function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Count successful and failed batches
    successful_batches = 0
    failed_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Validation")):
            try:
                # Extract batch data with error handling
                try:
                    images = batch['image'].cuda()
                    labels = batch['label'].cuda()
                    coords = batch.get('coords', None)
                    if coords is not None:
                        coords = coords.cuda()
                except Exception as e:
                    failed_batches += 1
                    if i % 50 == 0:
                        print(f"Val batch {i} error: {str(e)[:50]}")
                    continue
                
                outputs = model(images, coords) if coords is not None else model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                successful_batches += 1
                
            except Exception as e:
                failed_batches += 1
                if i % 50 == 0:
                    print(f"Val error batch {i}: {str(e)[:50]}")
                continue
    
    if successful_batches == 0:
        print("No successful validation batches!")
        return float('inf'), 0.0, {}
    
    epoch_loss = running_loss / successful_batches
    
    # Handle the case where we have no predictions
    if len(all_preds) == 0 or len(all_labels) == 0:
        print("No predictions collected during validation")
        return epoch_loss, 0.0, {}
    
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # Get detailed classification report with error handling
    try:
        # Limit to first 5 classes in console output to reduce spam
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=[val_dataset.id2label[i] for i in range(min(val_dataset.num_classes, 5))],
            output_dict=True
        )
    except Exception as e:
        print(f"Error in classification report: {str(e)[:50]}")
        report = {}
    
    print(f"Validation: {successful_batches} ok, {failed_batches} failed. Loss: {epoch_loss:.4f}, Acc: {accuracy:.4f}")
    return epoch_loss, accuracy, report

# Training loop
best_accuracy = 0.0
for epoch in range(config.num_epochs):
    # Training
    train_loss = train_epoch(
        model, 
        train_loader, 
        optimizer, 
        criterion, 
        scheduler, 
        scaler, 
        epoch
    )
    
    # Validation
    val_loss, accuracy, report = validate(model, val_loader, criterion)
    
    # Log metrics
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": accuracy,
        **{f"class_{k}_f1": v['f1-score'] for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
    })
    
    print(f"Epoch {epoch+1}/{config.num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_vit_landcover_model.pth")
        print(f"New best model saved with accuracy: {accuracy:.4f}")

# Save final model
torch.save({
    'epoch': config.num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_accuracy': best_accuracy,
    'id2label': val_dataset.id2label,
    'label2id': val_dataset.label2id
}, "final_vit_landcover_model.pth")

print(f"Training completed. Best accuracy: {best_accuracy:.4f}")

# Inference function for making predictions
def predict(model, image_path, transform, device="cuda"):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    
    return pred.item()