#!/usr/bin/env python3
"""
Script to download Sentinel-2 imagery for London using Google Earth Engine (GEE)
and save it locally.
"""

import ee
import os
import time
import requests
import argparse
from tqdm import tqdm
import osmnx as ox
import geopandas as gpd

# Initialize Google Earth Engine (GEE)
ee.Initialize()

def parse_args():
    parser = argparse.ArgumentParser(description='Download Sentinel-2 imagery for London')
    parser.add_argument('--output_dir', type=str, default='london_sentinel',
                        help='Directory to save imagery (default: london_sentinel)')
    parser.add_argument('--year', type=int, default=2023,
                        help='Year to filter imagery (default: 2023)')
    parser.add_argument('--cloud_pct', type=int, default=10,
                        help='Maximum cloud percentage (default: 10)')
    parser.add_argument('--tile_size', type=float, default=0.05,
                        help='Tile size in degrees (default: 0.05)')
    parser.add_argument('--bands', type=str, default='B4,B3,B2',
                        help='Bands to download (default: RGB)')
    return parser.parse_args()

def get_london_boundary():
    london_boundary = ox.geocode_to_gdf("London, UK")

    bounds = london_boundary.total_bounds 

    london_bbox = ee.Geometry.Rectangle(bounds.tolist())

    return ee.FeatureCollection([ee.Feature(london_bbox, {'name': 'London'})])


def get_london_borough_boundary(borough_name="Hammersmith and Fulham", gpkg_path="London_Boroughs.gpkg"):
    """
    Extract a specific London borough boundary from the Geopackage and convert to Earth Engine FeatureCollection.
    
    Args:
        borough_name (str): Exact name of the London borough (case-sensitive)
        gpkg_path (str): Path to the London Boroughs Geopackage file
        
    Returns:
        ee.FeatureCollection: Earth Engine FeatureCollection containing the borough boundary
    """
    import geopandas as gpd
    import ee
    import json
    
    # Ensure Earth Engine is initialized
    try:
        ee.Initialize()
    except:
        pass
    
    # Read the Geopackage file
    gdf = gpd.read_file(gpkg_path)
    
    # Filter for the specific borough - using exact match
    borough_gdf = gdf[gdf['name'] == borough_name]
    
    if borough_gdf.empty:
        raise ValueError(f"Borough '{borough_name}' not found. Please check spelling and case.")
    
    # Get the first feature (in case there are multiple matches)
    geometry = borough_gdf.iloc[0].geometry
    
    # Convert to GeoJSON-like dictionary
    geom_json = json.loads(gpd.GeoSeries([geometry]).to_json())
    coords = geom_json['features'][0]['geometry']['coordinates']
    
    # Create appropriate Earth Engine geometry based on the type
    geom_type = geom_json['features'][0]['geometry']['type']
    
    if geom_type == 'Polygon':
        ee_geometry = ee.Geometry.Polygon(coords)
    elif geom_type == 'MultiPolygon':
        ee_geometry = ee.Geometry.MultiPolygon(coords)
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")
    
    # Create a feature with borough name as property
    feature = ee.Feature(ee_geometry, {'name': borough_name})
    
    # Return as a FeatureCollection
    return ee.FeatureCollection([feature])


def download_tile(tile, composite, bands, output_dir):
    """Download a Sentinel-2 tile"""
    filename = f"london_{tile['lon']:.4f}_{tile['lat']:.4f}.tif"
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        print(f"Skipping {filename}, already downloaded.")
        return
    
    url = composite.select(bands.split(','))\
        .getDownloadURL({
            'scale': 10,
            'region': ee.Geometry.Rectangle(
                [tile['lon'], tile['lat'],
                 tile['lon'] + tile['width'], tile['lat'] + tile['height']]
            ),
            'format': 'GEO_TIFF'
        })
    
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    else:
        print(f"Failed to download {filename}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading Sentinel-2 imagery for London ({args.year}) with cloud cover < {args.cloud_pct}%")
    london_boundary = get_london_boundary()
    
    sentinel = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')\
        .filterBounds(london_boundary)\
        .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', args.cloud_pct))
    
    composite = sentinel.median()
    
    tiles = [{'lon': -0.5 + i * args.tile_size, 'lat': 51.2 + j * args.tile_size,
              'width': args.tile_size, 'height': args.tile_size}
             for i in range(int(0.8 / args.tile_size))
             for j in range(int(0.5 / args.tile_size))]
    
    for tile in tqdm(tiles, desc="Downloading tiles"):
        download_tile(tile, composite, args.bands, args.output_dir)
    
    print("Download complete!")

if __name__ == "__main__":
    main()