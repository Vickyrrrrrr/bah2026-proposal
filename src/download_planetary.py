import os
import sys
import requests
from pathlib import Path
from datetime import datetime, timedelta
from pystac_client import Client

# Auto-install planetary-computer if missing
try:
    import planetary_computer
except ImportError:
    print("📦 Installing planetary-computer library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "planetary-computer"])
    import planetary_computer

# Auto-install rasterio if missing
try:
    import rasterio
except ImportError:
    print("📦 Installing rasterio library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterio"])
    import rasterio

import time

def download_url(url, dest_path, retries=3, delay=5):
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)
            print(f"  ✅ Downloaded {dest_path.name}")
            return
        except Exception as e:
            print(f"  ⚠️ Download failed (Attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

def main():
    # Target Coordinates (Punjab agricultural crop lands)
    # [min_lon, min_lat, max_lon, max_lat]
    bbox = [74.8, 30.1, 74.9, 30.2]
    date_range = "2024-08-01/2024-08-30"
    out_dir = "./downloads_planetary"
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Connecting to Microsoft Planetary Computer STAC API...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    
    # Step 1: Search Sentinel-2 (Optical) for cloudy scene
    print("\n🔍 Step 1: Searching for a cloudy Sentinel-2 (Optical) scene...")
    search_s2 = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"gt": 30}}
    )
    s2_items = list(search_s2.get_items())
    if not s2_items:
        print("❌ No cloudy Sentinel-2 scenes found in the date range.")
        return
        
    target_s2 = s2_items[0]
    cloud_pct = target_s2.properties.get("eo:cloud_cover")
    print(f"✅ Found Sentinel-2 Scene: {target_s2.id} (Cloud cover: {cloud_pct:.1f}%)")
    
    # Step 2: Search Sentinel-1 (SAR) matching date
    acq_datetime = target_s2.properties.get("datetime")
    acq_date = acq_datetime[:10]
    print(f"🕒 S2 Acquisition date: {acq_date}")
    
    date_obj = datetime.strptime(acq_date, "%Y-%m-%d")
    start_date = (date_obj - timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=3)).strftime("%Y-%m-%d")
    sar_date_range = f"{start_date}/{end_date}"
    
    print(f"\n🔍 Step 2: Searching for matching Sentinel-1 SAR scene between {start_date} and {end_date}...")
    search_s1 = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=sar_date_range
    )
    s1_items = list(search_s1.get_items())
    if not s1_items:
        print("❌ No matching Sentinel-1 SAR scenes found.")
        return
        
    target_s1 = s1_items[0]
    print(f"✅ Found matching Sentinel-1 Scene: {target_s1.id}")
    
    # Sign items using Planetary Computer SAS token generator
    signed_s2 = planetary_computer.sign(target_s2)
    signed_s1 = planetary_computer.sign(target_s1)
    
    # Step 3: Download Sentinel-2 Green (B03), Red (B04), and NIR (B08)
    print("\n💾 Step 3: Downloading Sentinel-2 bands...")
    s2_files = {}
    for band in ["B03", "B04", "B08"]:
        asset = signed_s2.assets.get(band)
        url = asset.href
        dest_file = out_path / f"s2_{band}_temp.tif"
        s2_files[band] = dest_file
        
        if dest_file.exists():
            print(f"  File already exists: {dest_file.name}")
            continue
        print(f"  Downloading {band}...")
        download_url(url, dest_file)
        
    # Step 4: Download Sentinel-1 VV and VH
    print("\n💾 Step 4: Downloading Sentinel-1 SAR bands...")
    s1_files = {}
    for polar in ["vv", "vh"]:
        asset = signed_s1.assets.get(polar)
        url = asset.href
        dest_file = out_path / f"s1_{polar}_temp.tif"
        s1_files[polar] = dest_file
        
        if dest_file.exists():
            print(f"  File already exists: {dest_file.name}")
            continue
        print(f"  Downloading {polar}...")
        download_url(url, dest_file)
        
    # Step 5: Stack single-band files into multi-band TIFFs
    print("\n📦 Step 5: Stacking Sentinel-2 bands into 3-band GeoTIFF...")
    s2_stacked = out_path / f"s2_cloudy_stacked.tif"
    with rasterio.open(s2_files["B03"]) as src:
        meta = src.meta.copy()
        meta.update(count=3)
        
    with rasterio.open(s2_stacked, 'w', **meta) as dst:
        for idx, band in enumerate(["B03", "B04", "B08"]):
            with rasterio.open(s2_files[band]) as src:
                dst.write(src.read(1), idx + 1)
    print(f"  ✅ Created 3-band optical GeoTIFF: {s2_stacked.name}")
    
    print("\n📦 Step 6: Stacking Sentinel-1 bands into 2-band GeoTIFF...")
    s1_stacked = out_path / f"s1_sar_stacked.tif"
    with rasterio.open(s1_files["vv"]) as src:
        meta = src.meta.copy()
        meta.update(count=2)
        
    with rasterio.open(s1_stacked, 'w', **meta) as dst:
        for idx, polar in enumerate(["vv", "vh"]):
            with rasterio.open(s1_files[polar]) as src:
                dst.write(src.read(1), idx + 1)
    print(f"  ✅ Created 2-band SAR GeoTIFF: {s1_stacked.name}")
    
    # Clean up single band files
    print("\n🧹 Cleaning up temporary files...")
    for f in list(s2_files.values()) + list(s1_files.values()):
        if f.exists():
            os.remove(f)
            
    print(f"\n🎉 Success! All stacked files ready in: {out_path.resolve()}")
    print("👉 Optical Stacked: downloads_planetary/s2_cloudy_stacked.tif")
    print("👉 SAR Stacked:     downloads_planetary/s1_sar_stacked.tif")
    print("\n🚀 You can run predict.py on these files directly!")

if __name__ == "__main__":
    main()
