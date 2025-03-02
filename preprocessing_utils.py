import osmnx as ox
import geopandas as gpd
import ee
import os
from glob import glob
import rasterio
from rasterio.mask import mask
import numpy as np
from rasterio.merge import merge
import shutil

def get_central_london_boundary():
    # Get the boroughs of London
    london_boroughs = ox.geocode_to_gdf("London boroughs, UK")
    
    # Define Central London boroughs
    central_boroughs = [
        "Westminster", "Camden", "Islington", "Hackney", "Tower Hamlets", "Southwark", "Lambeth",
        "Kensington and Chelsea"
    ]
    
    # Filter for Central London boroughs only
    central_london = london_boroughs[london_boroughs["name"].isin(central_boroughs)]
    
    # Get the bounding box (minx, miny, maxx, maxy)
    bounds = central_london.total_bounds  # [minx, miny, maxx, maxy]
    
    # Convert bounds to an Earth Engine rectangle
    central_london_bbox = ee.Geometry.Rectangle(bounds.tolist())
    
    # Create a FeatureCollection with a name property
    return ee.FeatureCollection([ee.Feature(central_london_bbox, {'name': 'Central London'})])

def map_lidar_to_sentinel(lidar_file, sentinel_dir, output_dir, img_index):
    """
    Maps a single LiDAR image to all corresponding Sentinel-2 images based on spatial overlap,
    merges overlapping Sentinel images, and crops the final output to match the LiDAR extent.
    
    Parameters:
    - lidar_file (str): Path to the LiDAR TIFF image.
    - sentinel_dir (str): Directory containing Sentinel-2 TIFF images.
    - output_dir (str): Directory to save the final mapped Sentinel-2 image.
    - img_index (int): Index for naming files consistently.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        sentinel_files = glob(os.path.join(sentinel_dir, "*.tif"))
        print("Searching for ", lidar_file)
        
        with rasterio.open(lidar_file) as lidar_src:
            lidar_bounds = lidar_src.bounds
            print("LiDAR Bounds:", lidar_bounds)
            lidar_crs = lidar_src.crs
        
        matched_rasters = []
        
        # Find matching Sentinel-2 images
        for i, sentinel_file in enumerate(sentinel_files, start=1):
            print("Checking", sentinel_file)
            with rasterio.open(sentinel_file) as sentinel_src:
                sentinel_crs = sentinel_src.crs
                sentinel_bounds = sentinel_src.bounds
                
                # Ensure CRS matches
                if lidar_crs != sentinel_crs:
                    print("CRS mismatch, skipping", sentinel_file)
                    continue
                
                # Check if Sentinel-2 image overlaps with LiDAR image
                if (lidar_bounds.right > sentinel_bounds.left and
                    lidar_bounds.left < sentinel_bounds.right and
                    lidar_bounds.top > sentinel_bounds.bottom and
                    lidar_bounds.bottom < sentinel_bounds.top):

                    print("Found a match :D")
                    
                    # Crop Sentinel-2 image to LiDAR extent
                    bbox_geom = [{
                        "type": "Polygon",
                        "coordinates": [[
                            [lidar_bounds.left, lidar_bounds.bottom],
                            [lidar_bounds.right, lidar_bounds.bottom],
                            [lidar_bounds.right, lidar_bounds.top],
                            [lidar_bounds.left, lidar_bounds.top],
                            [lidar_bounds.left, lidar_bounds.bottom]
                        ]]
                    }]
                    
                    cropped_image, cropped_transform = mask(sentinel_src, bbox_geom, crop=True)
                    cropped_meta = sentinel_src.meta.copy()
                    cropped_meta.update({
                        "height": cropped_image.shape[1],
                        "width": cropped_image.shape[2],
                        "transform": cropped_transform
                    })
                    
                    # Ensure all images have the same number of bands
                    if len(cropped_image.shape) == 2:
                        cropped_image = np.expand_dims(cropped_image, axis=0)
                    
                    cropped_meta.update({"count": cropped_image.shape[0]})
                    
                    # Save cropped raster as a temporary file
                    temp_cropped_path = os.path.join(output_dir, f"temp_{i}.tif")
                    with rasterio.open(temp_cropped_path, "w", **cropped_meta) as temp_dst:
                        temp_dst.write(cropped_image)
                    
                    matched_rasters.append(temp_cropped_path)
        
        print("Merging matches")
        
        if not matched_rasters:
            print("No matching Sentinel images found for", lidar_file)
            return
        
        # Open the temporary rasters as DatasetReaders for merging
        try:
            src_files_to_merge = [rasterio.open(f) for f in matched_rasters]
            merged_data, merged_transform = merge(src_files_to_merge, method='first')
            merged_meta = src_files_to_merge[0].meta.copy()
            merged_meta.update({
                "height": merged_data.shape[1],
                "width": merged_data.shape[2],
                "transform": merged_transform,
                "count": merged_data.shape[0]  # Ensure correct band count
            })
            
            # Convert to expected format before saving
            merged_data = merged_data.astype(merged_meta["dtype"])
            
            # Save the merged raster temporarily for cropping
            temp_merged_path = os.path.join(output_dir, "temp_merged.tif")
            with rasterio.open(temp_merged_path, "w", **merged_meta) as temp_dst:
                temp_dst.write(merged_data)
            
            # Crop the merged raster to exactly match LiDAR bounds
            with rasterio.open(temp_merged_path) as temp_src:
                final_cropped_image, final_cropped_transform = mask(temp_src, bbox_geom, crop=True)
            
            merged_meta.update({
                "height": final_cropped_image.shape[1],
                "width": final_cropped_image.shape[2],
                "transform": final_cropped_transform,
                "count": final_cropped_image.shape[0]  # Ensure correct band count
            })
            
            # Save the final cropped Sentinel and LiDAR images with the correct naming convention
            lidar_output_path = os.path.join(output_dir, f"img_{img_index}_lidar.tif")
            sentinel_output_path = os.path.join(output_dir, f"img_{img_index}_sentinel.tif")
            
            with rasterio.open(sentinel_output_path, "w", **merged_meta) as dst:
                dst.write(final_cropped_image)
            
            shutil.copy(lidar_file, lidar_output_path)
        
        except Exception as merge_error:
            print(f"Error merging Sentinel images for {lidar_file}: {merge_error}")
            return
        
        # Close all dataset readers and remove temp files
        finally:
            for src in src_files_to_merge:
                src.close()
            os.remove(temp_merged_path)
            for temp_file in matched_rasters:
                os.remove(temp_file)
        
        print(f"Saved: {lidar_output_path} and {sentinel_output_path}")
    
    except Exception as e:
        print(f"Error processing {lidar_file}: {e}")
        return

# Iterate over all LiDAR files and pass an index
lidar_files = glob(os.path.join("lidar_data/reprojected", "*.tif"))
for img_index, lidar_file in enumerate(lidar_files, start=1):
    print(f"Processing LiDAR file {img_index}: {lidar_file}")
    map_lidar_to_sentinel(lidar_file, "london_sentinel", "final_dataset", img_index)