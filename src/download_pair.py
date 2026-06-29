import os
import argparse
from datetime import datetime, timedelta
from bhoonidhi_client import BhoonidhiClient

def main():
    parser = argparse.ArgumentParser(description="Download matching LISS-4 and Sentinel-1 SAR pair from Bhoonidhi")
    parser.add_argument("--username", type=str, required=True, help="Bhoonidhi userId/username")
    parser.add_argument("--password", type=str, required=True, help="Bhoonidhi password")
    parser.add_argument("--out_dir", type=str, default="./downloads", help="Output directory for downloads")
    args = parser.parse_args()

    # Initialize client
    client = BhoonidhiClient(username=args.username, password=args.password)
    if not client.token:
        print("❌ Authentication failed. Please check your credentials.")
        return
    
    # Punjab crop region coordinates [min_lon, min_lat, max_lon, max_lat]
    bbox = [74.5, 30.0, 75.0, 30.5] 
    date_range = "2024-08-01/2024-08-30" # Monsoon month
    
    print("\n🔍 Step 1: Searching for a cloudy LISS-4 multispectral scene...")
    liss4_scenes = client.search_scenes("Resourcesat2_LISS4", bbox, date_range, cloud_cover_gt=30)
    
    if not liss4_scenes:
        print("❌ No cloudy LISS-4 scenes found for these coordinates in August 2024.")
        return
        
    target_liss4 = liss4_scenes[0]
    cloud_pct = target_liss4.properties.get("eo:cloud_cover", "unknown")
    print(f"✅ Found LISS-4 Scene: {target_liss4.id} (Cloud cover: {cloud_pct}%)")
    
    # Get exact date of the LISS-4 scene to search matching radar
    acq_datetime = target_liss4.properties.get("datetime")
    acq_date = acq_datetime[:10]
    print(f"🕒 LISS-4 Acquisition date: {acq_date}")
    
    # Calculate a 3-day search window around LISS-4 date
    date_obj = datetime.strptime(acq_date, "%Y-%m-%d")
    start_date = (date_obj - timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=3)).strftime("%Y-%m-%d")
    sar_date_range = f"{start_date}/{end_date}"
    
    print(f"\n🔍 Step 2: Searching for matching Sentinel-1 SAR scene between {start_date} and {end_date}...")
    sar_scenes = client.search_scenes("Sentinel1_SAR", bbox, sar_date_range, cloud_cover_gt=0)
    
    if not sar_scenes:
        print("❌ No overlapping Sentinel-1 SAR scenes found within the 3-day window.")
        return
        
    target_sar = sar_scenes[0]
    print(f"✅ Found matching SAR Scene: {target_sar.id}")
    
    # Download both assets
    print("\n💾 Step 3: Downloading LISS-4 optical GeoTIFF...")
    client.download_asset(target_liss4, args.out_dir)
    
    print("\n💾 Step 4: Downloading Sentinel-1 radar GeoTIFF...")
    client.download_asset(target_sar, args.out_dir)
    
    print(f"\n🎉 Success! Files saved in: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
