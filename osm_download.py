#!/usr/bin/env python3
"""
Script to download the complete United Kingdom OpenStreetMap data from Geofabrik.
"""

import os
import requests
from tqdm import tqdm
import time

# UK OpenStreetMap data direct URL
UK_OSM_URL = "https://download.geofabrik.de/europe/united-kingdom-latest.osm.pbf"

def download_uk_osm(output_dir=".", filename="united-kingdom-latest.osm.pbf", max_retries=3):
    """
    Download the complete UK OpenStreetMap data file
    
    Parameters:
    -----------
    output_dir : str
        Directory where the file will be saved
    filename : str
        Name of the output file
    max_retries : int
        Maximum number of retry attempts
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    temp_path = output_path + ".tmp"
    
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"UK OSM file already exists at {output_path}")
        user_input = input("Do you want to download it again? (y/n): ").lower()
        if user_input != 'y':
            print("Download skipped.")
            return True
    
    retries = 0
    while retries < max_retries:
        try:
            # Send a GET request to the URL
            print(f"Downloading UK OpenStreetMap data from {UK_OSM_URL}")
            response = requests.get(UK_OSM_URL, stream=True, timeout=60)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(temp_path, 'wb') as f, tqdm(
                    desc="Downloading UK OSM data",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            
            # Rename temp file to final file
            os.rename(temp_path, output_path)
            
            # Verify file size
            downloaded_size = os.path.getsize(output_path)
            if total_size > 0 and downloaded_size >= total_size * 0.99:
                print(f"\nUK OpenStreetMap data downloaded successfully!")
                print(f"File saved to: {output_path}")
                print(f"File size: {downloaded_size / (1024*1024):.1f} MB")
                return True
            else:
                print(f"\nWarning: Downloaded file size ({downloaded_size} bytes) doesn't match expected size ({total_size} bytes)")
                retries += 1
                if retries < max_retries:
                    print(f"Retrying ({retries}/{max_retries})...")
                
        except requests.exceptions.RequestException as e:
            print(f"\nError downloading file: {e}")
            retries += 1
            if retries < max_retries:
                wait_time = 5 * retries
                print(f"Retrying in {wait_time} seconds... ({retries}/{max_retries})")
                time.sleep(wait_time)
        
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            retries += 1
            if retries < max_retries:
                wait_time = 5 * retries
                print(f"Retrying in {wait_time} seconds... ({retries}/{max_retries})")
                time.sleep(wait_time)
    
    print(f"\nFailed to download UK OpenStreetMap data after {max_retries} attempts.")
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download UK OpenStreetMap data from Geofabrik')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save OSM data (default: current directory)')
    parser.add_argument('--filename', type=str, default='united-kingdom-latest.osm.pbf',
                        help='Output filename (default: united-kingdom-latest.osm.pbf)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retry attempts (default: 3)')
    
    args = parser.parse_args()
    
    download_uk_osm(
        output_dir=args.output_dir,
        filename=args.filename,
        max_retries=args.max_retries
    )