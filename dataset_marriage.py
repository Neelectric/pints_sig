import os
import re
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from pyproj import Transformer
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, Features, Value, Array3D, Array2D, concatenate_datasets
from tqdm import tqdm
from skimage.transform import resize

def extract_coordinates(filename):
    """Extract longitude and latitude from Sentinel-2 filename."""
    match = re.search(r'_(-?\d+\.\d+)_(\d+\.\d+)\.tif$', filename)
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))
        return lon, lat
    else:
        raise ValueError(f"Could not extract coordinates from filename: {filename}")

class UKLandCoverDataset(Dataset):
    def __init__(self, sentinel_dir, land_cover_path, valid_images_file=None, cache_dir=None, transform=None):
        """
        Dataset for aligning Sentinel-2 images with UK land cover map.
        
        Args:
            sentinel_dir: Directory containing Sentinel-2 .tif files
            land_cover_path: Path to the land cover .tif file
            valid_images_file: Optional file containing list of valid image filenames
            cache_dir: Optional directory to cache extracted land cover regions
            transform: Optional transformations to apply to images
        """
        self.sentinel_dir = sentinel_dir
        self.land_cover_path = land_cover_path
        self.cache_dir = cache_dir
        self.transform = transform
        
        # Load list of valid images if provided, otherwise use all .tif files
        if valid_images_file and os.path.exists(valid_images_file):
            with open(valid_images_file, 'r') as f:
                self.sentinel_files = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(self.sentinel_files)} valid images from {valid_images_file}")
        else:
            self.sentinel_files = [f for f in os.listdir(sentinel_dir) if f.endswith('.tif')]
            print(f"Using all {len(self.sentinel_files)} images from directory")
        
        # Create cache directory if needed
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Initialize coordinate transformer (WGS84 -> British National Grid)
        self.wgs84_to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        
        # Open land cover map in read mode
        self.land_cover = rasterio.open(land_cover_path)
        
        # Store land cover metadata
        self.land_cover_transform = self.land_cover.transform
        self.land_cover_crs = self.land_cover.crs
        
        print(f"Initialized dataset with {len(self.sentinel_files)} Sentinel-2 images")
        print(f"Land cover map shape: {self.land_cover.height}x{self.land_cover.width}")

    def get_land_cover_window(self, lon, lat):
        """
        Convert WGS84 coordinates to a window in the land cover file.
        This function handles the specific bounds of the GB land cover map.
        """
        # First, transform the coordinates to BNG
        easting, northing = self.wgs84_to_bng.transform(lon, lat)
        
        # Print raw coordinates for debugging
        print(f"WGS84: {lon}, {lat} -> BNG: {easting}, {northing}")
        
        # Define the actual extent of the GB land cover map from the metadata
        # Based on your table: (0, 700000, 0, 1300000)
        gb_min_easting = 0
        gb_max_easting = 700000  
        gb_min_northing = 0
        gb_max_northing = 1300000
        
        # Check if the transformed coordinates are within the raster bounds
        if (easting < gb_min_easting or easting > gb_max_easting or
            northing < gb_min_northing or northing > gb_max_northing):
            print(f"Warning: Coordinates {easting}, {northing} are outside GB land cover map bounds")
            
            # Clip to the bounds of the land cover map
            easting = max(gb_min_easting, min(gb_max_easting, easting))
            northing = max(gb_min_northing, min(gb_max_northing, northing))
            print(f"Clipped to: {easting}, {northing}")
        
        # Now we need to check if this location actually has data in the raster
        # First, convert to pixel coordinates
        row, col = ~self.land_cover_transform * (easting, northing)
        row, col = int(row), int(col)
        
        # Check if these pixel coordinates are valid
        if (row < 0 or row >= self.land_cover.height or col < 0 or col >= self.land_cover.width):
            print(f"Warning: Pixel coordinates ({row}, {col}) are outside raster dimensions")
            # Clip to valid pixel coordinates
            row = max(0, min(self.land_cover.height - 1, row))
            col = max(0, min(self.land_cover.width - 1, col))
        
        # For a typical Sentinel-2 image at 10m resolution
        # Calculate window size in pixels (10km ÷ 10m = 1000 pixels)
        window_half_width = 500  # pixels (5km at 10m resolution)
        window_half_height = 500  # pixels (5km at 10m resolution)
        
        # Calculate window boundaries in pixel coordinates
        row_off = max(0, row - window_half_height)
        row_end = min(self.land_cover.height, row + window_half_height)
        col_off = max(0, col - window_half_width)
        col_end = min(self.land_cover.width, col + window_half_width)
        
        # Ensure we have a valid window size
        if row_end - row_off < 5:
            row_end = min(row_off + 5, self.land_cover.height)
        if col_end - col_off < 5:
            col_end = min(col_off + 5, self.land_cover.width)
        
        window = ((row_off, row_end), (col_off, col_end))
        print(f"Land cover window in pixels: {window}")
        
        # Calculate the actual size of the window in meters
        window_width_meters = (col_end - col_off) * 10  # 10m per pixel
        window_height_meters = (row_end - row_off) * 10  # 10m per pixel
        print(f"Window size: {window_width_meters}m × {window_height_meters}m")
        
        return window

    
    def __len__(self):
        return len(self.sentinel_files)
    
    def __getitem__(self, idx):
        """Get a single image-label pair."""
        try:
            filename = self.sentinel_files[idx]
            filepath = os.path.join(self.sentinel_dir, filename)
            
            # Check if we have a cached label
            label_path = None
            if self.cache_dir:
                label_path = os.path.join(self.cache_dir, f"{os.path.splitext(filename)[0]}_label.npy")
                
            # Extract coordinates from filename
            lon, lat = extract_coordinates(filename)
            
            # Load Sentinel-2 image
            with rasterio.open(filepath) as src:
                image = src.read()  # Shape: (channels, height, width)
                img_height, img_width = image.shape[1], image.shape[2]
                
                # Check for unexpected dimensions and pad/crop if needed
                if image.shape != (3, 1154, 718):
                    # Create a properly sized array
                    padded_image = np.zeros((3, 1154, 718), dtype=image.dtype)
                    
                    # Copy data with size limits
                    h = min(image.shape[1], 1154)
                    w = min(image.shape[2], 718)
                    padded_image[:, :h, :w] = image[:, :h, :w]
                    image = padded_image
            
            # Try to load cached label if it exists
            if label_path and os.path.exists(label_path):
                label = np.load(label_path)
                
                # Ensure label has the right dimensions
                if label.shape != (1154, 718):
                    padded_label = np.zeros((1154, 718), dtype=label.dtype)
                    h = min(label.shape[0], 1154)
                    w = min(label.shape[1], 718)
                    padded_label[:h, :w] = label[:h, :w]
                    label = padded_label
            else:
                # Get window in land cover map
                window = self.get_land_cover_window(lon, lat)
                
                # Read the land cover labels
                try:
                    label = self.land_cover.read(1, window=window)
                    
                    # Resize to target size
                    label = resize(label, (1154, 718), 
                                 order=0,  # Nearest neighbor interpolation
                                 preserve_range=True).astype(np.int64)
                    
                    # Cache the label if requested
                    if label_path:
                        np.save(label_path, label)
                        
                except Exception as e:
                    print(f"Error reading land cover: {e}")
                    pass
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image.astype(np.float32),  # Ensure correct dtype
                'label': label,
                'coordinates': {'longitude': lon, 'latitude': lat},
                'filename': filename
            }
                
        except Exception as e:
            # Return a dummy sample on error
            print(f"Error processing index {idx}, file: {self.sentinel_files[idx] if idx < len(self.sentinel_files) else 'unknown'}: {e}")
            return {
                'image': np.zeros((3, 1154, 718), dtype=np.float32),
                'label': np.zeros((1154, 718), dtype=np.int64),
                'coordinates': {'longitude': 0.0, 'latitude': 0.0},
                'filename': f"error_{idx}",
                'error': str(e)
            }
    
    def close(self):
        """Close the land cover file."""
        if hasattr(self, 'land_cover') and self.land_cover:
            self.land_cover.close()
    
    def __del__(self):
        """Ensure resources are freed."""
        self.close()

def create_huggingface_dataset(uk_dataset, sample_count=None, output_path=None, batch_size=50):
    """
    Convert a UKLandCoverDataset to a HuggingFace Dataset, processing in batches to manage memory.
    """
    # Create output directory if needed
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Define features with fixed dimensions
    features = Features({
        'image': Array3D(dtype='float32', shape=(3, 1154, 718)),
        'label': Array2D(dtype='int64', shape=(1154, 718)),
        'longitude': Value('float'),
        'latitude': Value('float'),
        'filename': Value('string'),
        'is_valid': Value('bool')  # Flag for valid samples
    })
    
    # Initialize empty dataset
    hf_dataset = None
    
    # Process in batches
    total = sample_count if sample_count else len(uk_dataset)
    total = min(total, len(uk_dataset))
    
    for batch_start in tqdm(range(0, total, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total)
        
        # Prepare batch data
        data_dict = {
            'image': [],
            'label': [],
            'longitude': [],
            'latitude': [],
            'filename': [],
            'is_valid': []
        }
        
        # Process batch
        for i in tqdm(range(batch_start, batch_end), desc=f"Processing samples {batch_start}-{batch_end}"):
            try:
                sample = uk_dataset[i]
                
                # Check if this is a valid sample
                is_valid = not (sample.get('filename', '').startswith('error_') or 
                               (sample.get('coordinates', {}).get('longitude', 0) == 0.0 and 
                                sample.get('coordinates', {}).get('latitude', 0) == 0.0) and
                                # "label" is a dummy array of zeros
                                np.all(sample.get('label', np.zeros((1154, 718), dtype=np.int64)) == 0)
                                )
                # also make sure that sample['label'] is not all zeros
                if np.all(sample.get('label', np.zeros((1154, 718), dtype=np.int64)) == 0):
                    is_valid = False
                
                data_dict['image'].append(sample['image'])
                data_dict['label'].append(sample['label'])
                data_dict['longitude'].append(sample['coordinates']['longitude'])
                data_dict['latitude'].append(sample['coordinates']['latitude'])
                data_dict['filename'].append(sample['filename'])
                data_dict['is_valid'].append(is_valid)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Add a dummy sample
                data_dict['image'].append(np.zeros((3, 1154, 718), dtype=np.float32))
                data_dict['label'].append(np.zeros((1154, 718), dtype=np.int64))
                data_dict['longitude'].append(0.0)
                data_dict['latitude'].append(0.0)
                data_dict['filename'].append(f"error_{i}")
                data_dict['is_valid'].append(False)
        
        # Create batch dataset
        batch_dataset = HFDataset.from_dict(data_dict, features=features)
        
        # Save intermediate batch before attempting to concatenate
        if output_path:
            batch_path = f"{output_path}_batch_{batch_start}_{batch_end}"
            batch_dataset.save_to_disk(batch_path)
            print(f"Saved batch to {batch_path}")
        
        # Merge with existing dataset or create new one
        if hf_dataset is None:
            hf_dataset = batch_dataset
        else:
            try:
                hf_dataset = concatenate_datasets([hf_dataset, batch_dataset])
            except Exception as e:
                print(f"Error concatenating datasets: {e}")
                print("Will continue with previously processed batches")
                # The batches are already saved individually, so we can recover later
    
    # Filter out invalid samples
    if hf_dataset is not None:
        valid_dataset = hf_dataset.filter(lambda x: x['is_valid'], num_proc=50)
        print(f"Total samples: {len(hf_dataset)}, Valid samples: {len(valid_dataset)}")
        
        # Save final dataset
        if output_path:
            valid_dataset.save_to_disk(output_path)
            print(f"Final filtered dataset saved to {output_path}")
        
        return valid_dataset
    else:
        print("No valid samples found")
        return None

# Example usage
if __name__ == "__main__":
    # Paths
    sentinel_dir = "./gee"
    land_cover_path = "/home/user/.cache/huggingface/hub/datasets--pints-sig--Land_Cover_Map_2023_10m_classified_pixels_GB/snapshots/93682d97808ea967159f6372d1805723fdc92b29/FME_61343034_1740838768029_7621/data/7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif"
    valid_images_file = "valid_images.txt"
    cache_dir = "./label_cache"
    output_path = "./uk_landcover_dataset"
    
    # Check if we have a list of valid images, otherwise scan for them
    if not os.path.exists(valid_images_file):
        print("No valid images list found. Run scan_valid_images.py first to identify valid images.")
        exit(1)
    
    # Create dataset using valid images
    uk_dataset = UKLandCoverDataset(
        sentinel_dir=sentinel_dir,
        land_cover_path=land_cover_path,
        valid_images_file=valid_images_file,
        cache_dir=cache_dir
    )
    
    print("Creating full dataset...")
    full_dataset = create_huggingface_dataset(
        uk_dataset=uk_dataset,
        output_path=output_path,
        batch_size=100
    )
    print("Dataset creation complete.")
    
    # Close dataset to free resources
    uk_dataset.close()