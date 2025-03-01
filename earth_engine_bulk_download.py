#!/usr/bin/env python3
"""
OPTIMIZED script to download 10m Sentinel-2 imagery for the UK using Google Earth Engine Python API
and save it directly to a high-performance cluster.
"""

import ee
import os
import time
import json
import datetime
import requests
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import argparse
import concurrent.futures
from functools import partial
import threading

# Initialize Earth Engine
ee.Initialize()

def parse_args():
    parser = argparse.ArgumentParser(description='Download UK satellite imagery to cluster')
    parser.add_argument('--output_dir', type=str, default='gee',
                        help='Relative directory to save imagery (default: "gee" in current directory)')
    parser.add_argument('--year', type=int, default=2023,
                        help='Year to filter imagery (default: 2023)')
    parser.add_argument('--cloud_pct', type=int, default=20,
                        help='Maximum cloud percentage (default: 20)')
    parser.add_argument('--num_processes', type=int, default=48,  # Increased from 24 to 48
                        help='Number of parallel download processes (default: 48)')
    parser.add_argument('--tile_size', type=float, default=0.1,
                        help='Size of each tile in degrees (default: 0.1)')
    parser.add_argument('--crs', type=str, default='EPSG:27700',
                        help='Coordinate reference system (default: EPSG:27700)')
    parser.add_argument('--bands', type=str, default='B4,B3,B2',
                        help='Comma-separated list of bands to download (default: B4,B3,B2)')
    parser.add_argument('--region', type=str, default='all',
                        help='Region to download: "all", "england", "scotland", "wales", or "northern_ireland" (default: all)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip tiles that have already been downloaded (default: True)')
    parser.add_argument('--force_redownload', action='store_true',
                        help='Force redownload of all tiles, even if they exist (default: False)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume downloading from where the script left off (default: False)')
    parser.add_argument('--wait_time', type=int, default=0,  # Reduced from 2 to 0
                        help='Wait time in seconds between batches (default: 0)')
    parser.add_argument('--batch_size', type=int, default=400,  # Increased from 100 to 400
                        help='Number of tiles to process before pausing (default: 400)')
    parser.add_argument('--max_retries', type=int, default=2,  # Reduced from 3 to 2
                        help='Maximum number of retries for failed downloads (default: 2)')
    parser.add_argument('--timeout', type=int, default=20,  # Reduced from 30 to 20
                        help='Timeout in seconds for each download request (default: 20)')
    parser.add_argument('--max_connections', type=int, default=100,
                        help='Maximum number of concurrent HTTP connections (default: 100)')
    parser.add_argument('--use_threadpool', action='store_true',
                        help='Use ThreadPoolExecutor instead of ProcessPool for better performance (default: False)')
    parser.add_argument('--aggressive', action='store_true',
                        help='Enable aggressive download mode with optimal settings (default: False)')
    return parser.parse_args()

def set_aggressive_mode(args):
    """Apply aggressive settings to maximize download speed"""
    if args.aggressive:
        args.num_processes = 54  # Very high parallelism
        args.wait_time = 0  # No waiting between batches
        args.batch_size = 800  # Large batches
        args.max_retries = 1  # Minimal retries
        args.timeout = 15  # Short timeout
        args.max_connections = 200  # Many connections
        args.use_threadpool = True  # Use threading instead of multiprocessing
    return args

def mask_s2_clouds(image):
    """Mask clouds in Sentinel-2 imagery using QA band"""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def get_uk_boundary():
    """Get UK boundary as Earth Engine feature collection"""
    # Using the LSIB dataset which contains country boundaries
    uk = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
        .filter(ee.Filter.eq('country_na', 'United Kingdom'))
    
    # Alternative method using World Atlas dataset if LSIB fails
    if uk.size().getInfo() == 0:
        uk = ee.FeatureCollection('projects/sat-io/open-datasets/geoboundaries/CGAZ_ADM0') \
            .filter(ee.Filter.eq('shapeName', 'United Kingdom'))
    
    # Fallback to manual definition if both datasets fail
    if uk.size().getInfo() == 0:
        uk_coords = [
            [-5.5, 49.9], [1.8, 49.9], [1.8, 61.0], [-5.5, 61.0], [-5.5, 49.9]
        ]
        uk = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Polygon([uk_coords]), {'name': 'United Kingdom'})
        ])
    
    return uk

def get_region_boundary(region_name):
    """Get boundary for a specific UK region"""
    if region_name == 'all':
        return get_uk_boundary()
    
    # Approximate bounding boxes for UK regions
    regions = {
        'england': {
            'min_lon': -5.7, 'max_lon': 1.8,
            'min_lat': 49.9, 'max_lat': 55.8
        },
        'scotland': {
            'min_lon': -8.6, 'max_lon': -0.7,
            'min_lat': 54.6, 'max_lat': 60.9
        },
        'wales': {
            'min_lon': -5.3, 'max_lon': -2.8,
            'min_lat': 51.3, 'max_lat': 53.4
        },
        'northern_ireland': {
            'min_lon': -8.2, 'max_lon': -5.4,
            'min_lat': 54.0, 'max_lat': 55.3
        }
    }
    
    if region_name not in regions:
        print(f"Region {region_name} not found. Using full UK boundary instead.")
        return get_uk_boundary()
    
    region = regions[region_name]
    coords = [
        [region['min_lon'], region['min_lat']],
        [region['max_lon'], region['min_lat']],
        [region['max_lon'], region['max_lat']],
        [region['min_lon'], region['max_lat']],
        [region['min_lon'], region['min_lat']]
    ]
    
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Polygon([coords]), {'name': region_name})
    ])

def create_grid_from_bounds(min_lon, max_lon, min_lat, max_lat, tile_size):
    """Create a grid of tiles from explicit boundary coordinates"""
    # Create grid tiles
    lon_grid = np.arange(min_lon, max_lon + tile_size, tile_size)
    lat_grid = np.arange(min_lat, max_lat + tile_size, tile_size)
    
    tiles = []
    for lon in lon_grid[:-1]:
        for lat in lat_grid[:-1]:
            tiles.append({
                'lon': lon,
                'lat': lat,
                'width': tile_size,
                'height': tile_size
            })
    
    return tiles

def create_grid(uk_bounds, tile_size):
    """Create a grid of tiles to download"""
    try:
        # Try to get coordinates from the bounds geometry
        coords = uk_bounds.getInfo()['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
    except (KeyError, IndexError) as e:
        print(f"Warning: Error getting coordinates from bounds: {e}")
        print("Using fallback boundary coordinates for the UK")
        # Fallback to approximate UK bounding box
        min_lon, max_lon = -9.0, 2.0
        min_lat, max_lat = 49.0, 61.0
        return create_grid_from_bounds(min_lon, max_lon, min_lat, max_lat, tile_size)
    
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    return create_grid_from_bounds(min_lon, max_lon, min_lat, max_lat, tile_size)

# Session factory with connection pooling
def create_session(max_connections):
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_connections,
        pool_maxsize=max_connections,
        max_retries=0  # We handle retries manually
    )
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def get_url_for_tile(tile, composite, bands, crs):
    """Get the download URL for a tile"""
    # Create a geometry for this tile
    geometry = ee.Geometry.Rectangle(
        [tile['lon'], tile['lat'], 
         tile['lon'] + tile['width'], tile['lat'] + tile['height']]
    )
    
    # Get the URL for the tile
    url = composite.select(bands.split(',')) \
        .getDownloadURL({
            'scale': 10,  # 10m resolution
            'crs': crs,
            'region': geometry,
            'format': 'GEO_TIFF',
            'maxPixels': 1e8  # Limit to avoid request size errors
        })
    
    return url

def download_with_session(url, output_path, temp_path, session, timeout):
    """Download a file using the provided session"""
    try:
        response = session.get(url, stream=True, timeout=timeout)
        
        if response.status_code == 200:
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            
            # Move the temp file to the final location once download is complete
            os.rename(temp_path, output_path)
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

def download_tile(tile, composite, bands, output_dir, crs, force_redownload, timeout, max_retries, session=None):
    """Download a single tile of imagery using connection pooling"""
    # Create a filename based on the tile coordinates
    filename = f"uk_sentinel2_10m_{tile['lon']:.4f}_{tile['lat']:.4f}.tif"
    output_path = os.path.join(output_dir, filename)
    
    # Skip if file already exists and is valid (and we're not forcing redownload)
    if not force_redownload and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    
    # Use a "temp" file for downloading to prevent partial downloads
    temp_path = output_path + ".tmp"
    
    # Create a new session if one wasn't provided
    local_session = session if session else requests.Session()
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            # Get the URL for the tile
            url = get_url_for_tile(tile, composite, bands, crs)
            
            # Download the tile with a timeout
            success = download_with_session(url, output_path, temp_path, local_session, timeout)
            
            if success:
                return True
            elif attempt < max_retries - 1:
                time.sleep(0.5)  # Short pause before retry
                continue
            else:
                return False
        except ee.EEException as e:
            if "Total request size" in str(e):
                # If tile is too large, immediately split it into 4 smaller tiles
                smaller_width = tile['width'] / 2
                smaller_height = tile['height'] / 2
                
                # Create 4 smaller tiles
                smaller_tiles = [
                    {'lon': tile['lon'], 'lat': tile['lat'], 
                     'width': smaller_width, 'height': smaller_height},
                    {'lon': tile['lon'] + smaller_width, 'lat': tile['lat'], 
                     'width': smaller_width, 'height': smaller_height},
                    {'lon': tile['lon'], 'lat': tile['lat'] + smaller_height, 
                     'width': smaller_width, 'height': smaller_height},
                    {'lon': tile['lon'] + smaller_width, 'lat': tile['lat'] + smaller_height, 
                     'width': smaller_width, 'height': smaller_height}
                ]
                
                # Try to download each smaller tile
                results = []
                for st in smaller_tiles:
                    result = download_tile(st, composite, bands, output_dir, crs, 
                                          force_redownload, timeout, max_retries, local_session)
                    results.append(result)
                
                return all(results)
            elif attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                return False
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            else:
                return False
    
    return False

def download_tile_wrapper(args):
    """Wrapper function to unpack arguments for download_tile"""
    return download_tile(*args)

# Progress monitoring in a separate thread
def progress_monitor(progress_file, total_tiles, completed_event, progress_queue):
    """Monitor and update progress in a separate thread"""
    start_time = time.time()
    last_completed = 0
    
    while not completed_event.is_set():
        # Get current completed count
        completed = 0
        while not progress_queue.empty():
            completed += progress_queue.get()
        
        # Calculate download rate
        elapsed = time.time() - start_time
        completed_tiles = last_completed + completed
        download_rate = completed_tiles / elapsed if elapsed > 0 else 0
        
        # Update progress file
        with open(progress_file, 'w') as f:
            json.dump({
                'total_tiles': total_tiles,
                'completed_tiles': completed_tiles,
                'remaining_tiles': total_tiles - completed_tiles,
                'download_rate': f"{download_rate:.2f} tiles/sec",
                'estimated_time_remaining': f"{(total_tiles - completed_tiles) / download_rate / 60:.2f} minutes" if download_rate > 0 else "unknown",
                'elapsed_time': f"{elapsed / 60:.2f} minutes",
                'last_updated': datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        # Update last completed count
        last_completed = completed_tiles
        
        # Sleep for a short while
        time.sleep(5)

def main():
    args = parse_args()
    
    # Apply aggressive settings if requested
    args = set_aggressive_mode(args)
    
    # Check if output directory exists, if not create it
    output_dir = os.path.join(os.getcwd(), args.output_dir)
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Use the absolute path for all operations
    args.output_dir = output_dir
    
    print(f"Downloading UK Sentinel-2 imagery for {args.year} with cloud cover < {args.cloud_pct}%")
    print(f"Region: {args.region}, Tile size: {args.tile_size} degrees")
    print(f"Bands: {args.bands}")
    print(f"Using {'aggressive' if args.aggressive else 'standard'} download mode")
    print(f"Concurrent downloads: {args.num_processes}, Batch size: {args.batch_size}")
    
    # Get specified region boundary
    region_boundary = get_region_boundary(args.region)
    region_bounds = region_boundary.geometry().bounds()
    
    # Get Sentinel-2 collection - using the updated collection name
    start_date = f"{args.year}-01-01"
    end_date = f"{args.year}-12-31"
    
    # Try the new Sentinel-2 collection first (harmonized)
    try:
        sentinel = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterBounds(region_boundary) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', args.cloud_pct))
        
        # Check if the collection has images
        if sentinel.size().getInfo() == 0:
            raise Exception("No images found in COPERNICUS/S2_HARMONIZED")
    except Exception as e:
        print(f"Warning: {e}. Falling back to COPERNICUS/S2_SR")
        # Fall back to the deprecated collection if necessary
        sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(region_boundary) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', args.cloud_pct))
    
    # Apply cloud mask
    filtered = sentinel.map(mask_s2_clouds)
    
    # Create a median composite
    composite = filtered.median()
    
    # Create a grid of tiles to download
    tiles = create_grid(region_bounds, args.tile_size)
    print(f"Created {len(tiles)} tiles to download")
    
    # Handle existing files logic
    if args.resume or (not args.force_redownload):
        # Check which tiles are already downloaded
        existing_files = os.listdir(args.output_dir)
        existing_tiles = []
        for file in existing_files:
            if file.startswith("uk_sentinel2_10m_") and file.endswith(".tif"):
                # Extract coordinates from filename
                parts = file.replace("uk_sentinel2_10m_", "").replace(".tif", "").split("_")
                if len(parts) == 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        existing_tiles.append((lon, lat))
                    except ValueError:
                        continue
        
        print(f"Found {len(existing_tiles)} existing tiles")
        
        if not args.force_redownload:
            # Filter out tiles that have already been downloaded
            tiles_to_download = []
            for tile in tiles:
                key = (tile['lon'], tile['lat'])
                if key not in existing_tiles:
                    tiles_to_download.append(tile)
            
            print(f"Remaining tiles to download: {len(tiles_to_download)}")
            tiles = tiles_to_download
    else:
        print("Force redownload is enabled. Will download all tiles.")
    
    if len(tiles) == 0:
        print("All tiles already downloaded. Exiting.")
        return
    
    # Save progress file
    progress_file = os.path.join(args.output_dir, "download_progress.json")
    with open(progress_file, 'w') as f:
        json.dump({
            'total_tiles': len(tiles),
            'remaining_tiles': len(tiles),
            'region': args.region,
            'year': args.year,
            'tile_size': args.tile_size,
            'last_updated': datetime.datetime.now().isoformat()
        }, f, indent=2)
    
    # Create a shared session for all downloads
    session = create_session(args.max_connections)
    
    # Prepare download arguments
    download_args = [(tile, composite, args.bands, args.output_dir, args.crs, 
                      args.force_redownload, args.timeout, args.max_retries, session) 
                      for tile in tiles]
    
    # Set up progress monitoring
    completed_event = threading.Event()
    progress_queue = mp.Manager().Queue()
    
    # Start progress monitor thread
    monitor_thread = threading.Thread(
        target=progress_monitor,
        args=(progress_file, len(tiles), completed_event, progress_queue)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # Choose between thread pool and process pool
        if args.use_threadpool:
            # Use ThreadPoolExecutor for HTTP-bound I/O tasks
            print(f"Using ThreadPoolExecutor with {args.num_processes} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_processes) as executor:
                # Process in batches
                for i in range(0, len(download_args), args.batch_size):
                    batch = download_args[i:i+args.batch_size]
                    
                    print(f"Processing batch {i//args.batch_size + 1}/{(len(download_args) + args.batch_size - 1)//args.batch_size}")
                    
                    futures = [executor.submit(download_tile_wrapper, arg) for arg in batch]
                    
                    # Wait for batch to complete and track successes
                    batch_successes = 0
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(batch)):
                        if future.result():
                            batch_successes += 1
                    
                    # Add successes to progress queue
                    progress_queue.put(batch_successes)
                    
                    # Brief pause between batches if configured
                    if args.wait_time > 0 and i + args.batch_size < len(download_args):
                        print(f"Pausing for {args.wait_time} seconds...")
                        time.sleep(args.wait_time)
        else:
            # Use ProcessPoolExecutor for CPU-bound tasks
            print(f"Using ProcessPoolExecutor with {args.num_processes} workers")
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_processes) as executor:
                # Process in batches
                for i in range(0, len(download_args), args.batch_size):
                    batch = download_args[i:i+args.batch_size]
                    
                    print(f"Processing batch {i//args.batch_size + 1}/{(len(download_args) + args.batch_size - 1)//args.batch_size}")
                    
                    # Need to recreate session for each process, so don't pass in global session
                    batch_args = [(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], None) for arg in batch]
                    futures = [executor.submit(download_tile_wrapper, arg) for arg in batch_args]
                    
                    # Wait for batch to complete and track successes
                    batch_successes = 0
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(batch)):
                        if future.result():
                            batch_successes += 1
                    
                    # Add successes to progress queue
                    progress_queue.put(batch_successes)
                    
                    # Brief pause between batches if configured
                    if args.wait_time > 0 and i + args.batch_size < len(download_args):
                        print(f"Pausing for {args.wait_time} seconds...")
                        time.sleep(args.wait_time)
    
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Progress has been saved.")
        # Update progress file on interrupt is handled by the monitor thread
    finally:
        # Signal monitor thread to exit
        completed_event.set()
        monitor_thread.join(timeout=1.0)
    
    # Final status update
    print(f"\nDownload process completed. Check {progress_file} for final stats.")

if __name__ == "__main__":
    main()