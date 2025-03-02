import os
import rasterio
from glob import glob
from preprocessing_utils import map_lidar_to_sentinel

# Directories
lidar_path = "lidar_data/reprojected"
sentinel_dir = "london_sentinel"

# Get the first LiDAR file
lidar_files = glob(os.path.join(lidar_path, "*.tif"))
sat_files = glob(os.path.join(sentinel_dir, "*.tif"))

for img_index, lidar_file in enumerate(lidar_files, start=1):
    print(f"Processing LiDAR file {img_index}: {lidar_file}")
    map_lidar_to_sentinel(lidar_file, "london_sentinel", "final_dataset", img_index)