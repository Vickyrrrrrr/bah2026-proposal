import os
import argparse
import numpy as np
import torch
import rasterio
from PIL import Image

# Local imports
from model import LISS4ClearNet

def parse_args():
    parser = argparse.ArgumentParser(description="LISS-4 ClearNet Single Image Cloud Removal Inference")
    parser.add_argument("--cloudy_optical", type=str, required=True, help="Path to cloudy LISS-4 GeoTIFF (Green, Red, NIR)")
    parser.add_argument("--sar_radar", type=str, required=True, help="Path to Sentinel-1 SAR GeoTIFF (VV, VH)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--output_tif", type=str, default="./results/output_clear.tif", help="Path to save georeferenced GeoTIFF output")
    parser.add_argument("--output_png", type=str, default="./results/output_clear.png", help="Path to save visual false-color PNG preview")
    parser.add_argument("--crop_size", type=int, default=256, help="Dimensions of the crop window (must be multiple of 32)")
    parser.add_argument("--offset_y", type=int, default=-1, help="Starting Y pixel index for the crop window (defaults to center)")
    parser.add_argument("--offset_x", type=int, default=-1, help="Starting X pixel index for the crop window (defaults to center)")
    parser.add_argument("--num_res_blocks", type=int, default=-1, help="Number of residual blocks (auto-detected if omitted)")
    parser.add_argument("--channel_dim", type=int, default=-1, help="Channel dimension (auto-detected if omitted)")
    return parser.parse_args()

def save_rgb_false_color(tensor_chw, path):
    """
    Saves a 3-band array (Green, Red, NIR) as a standard False Color Infrared (FCIR) PNG.
    NIR -> Red channel, Red -> Green channel, Green -> Blue channel.
    """
    arr = tensor_chw.numpy() if hasattr(tensor_chw, 'numpy') else tensor_chw
    fcir = np.stack([arr[2], arr[1], arr[0]], axis=-1)
    fcir = np.clip(fcir, 0.0, 1.0)
    img_uint8 = (fcir * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)
    print(f"Saved visual preview to: {path}")

def run_inference():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running single-image inference on: {device}")
    
    # 1. Read metadata and calculate crop window
    with rasterio.open(args.cloudy_optical) as src_opt:
        opt_meta = src_opt.meta.copy()
        h_opt, w_opt = src_opt.height, src_opt.width
        opt_transform = src_opt.transform
        
        # Determine crop offsets
        if args.offset_y >= 0:
            start_h = args.offset_y
        else:
            start_h = max(0, (h_opt - args.crop_size) // 2)
            
        if args.offset_x >= 0:
            start_w = args.offset_x
        else:
            start_w = max(0, (w_opt - args.crop_size) // 2)
            
        # Ensure dimensions don't exceed source
        start_h = min(start_h, h_opt - args.crop_size)
        start_w = min(start_w, w_opt - args.crop_size)
        
        print(f"Applying crop window: size={args.crop_size} at offset=({start_h}, {start_w})")
        
        # Read only the crop window from disk (memory efficient)
        from rasterio.windows import Window
        win = Window(start_w, start_h, args.crop_size, args.crop_size)
        opt_data = src_opt.read([1, 2, 3], window=win).astype(np.float32)
        
        # Update metadata for cropped geotransform
        cropped_transform = rasterio.windows.transform(win, opt_transform)
        opt_meta.update({
            "height": args.crop_size,
            "width": args.crop_size,
            "transform": cropped_transform,
            "count": 3,
            "driver": "GTiff"
        })
        
    with rasterio.open(args.sar_radar) as src_sar:
        # Match crop window size
        sar_data = src_sar.read([1, 2], window=win).astype(np.float32)
        
    # 2. Preprocess / Normalize data
    print("Normalizing input data...")
    # Normalize optical (reflectance scale / 3000.0)
    opt_data = np.clip(opt_data / 3000.0, 0.0, 1.0)
    
    # Calibrate SAR (If raw amplitude is loaded, convert to dB, then scale)
    if sar_data.max() > 10.0:
        print("Calibrating raw SAR amplitude values to decibels...")
        db_vv = 20 * np.log10(sar_data[0] + 1e-5) - 58.0
        db_vh = 20 * np.log10(sar_data[1] + 1e-5) - 58.0
        sar_db = np.stack([db_vv, db_vh], axis=0)
        sar_db = np.clip(sar_db, -25.0, 0.0)
        sar_data = sar_db / 100.0
    else:
        # standard scaling if pre-converted to dB
        sar_data = np.clip(sar_data / 10000.0, -1.0, 1.0)
        
    cloudy_tensor = torch.from_numpy(opt_data).unsqueeze(0).to(device)
    sar_tensor = torch.from_numpy(sar_data).unsqueeze(0).to(device)
    
    # 3. Load Model Specs and weights
    print("Loading checkpoint weights...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Auto-detect architecture if not specified
    if args.channel_dim < 0:
        args.channel_dim = ckpt['opt_conv.0.weight'].shape[0]
    if args.num_res_blocks < 0:
        res_keys = [k for k in ckpt.keys() if 'res_loop' in k and 'weight' in k]
        args.num_res_blocks = len(res_keys) // 4
        
    print(f"Model specifications: channel_dim={args.channel_dim}, num_res_blocks={args.num_res_blocks}")
    
    model = LISS4ClearNet(num_res_blocks=args.num_res_blocks, channel_dim=args.channel_dim).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    
    # 4. Predict
    print("Running model forward pass...")
    with torch.no_grad():
        pred_tensor = model(cloudy_tensor, sar_tensor)
        pred_np = pred_tensor.cpu().squeeze(0).numpy()
        # Scale back to target reflectance range
        pred_scaled = (pred_np * 3000.0).astype(opt_meta['dtype'])
        
    # 5. Write Georeferenced GeoTIFF
    os.makedirs(os.path.dirname(args.output_tif), exist_ok=True)
    print(f"Writing georeferenced output: {args.output_tif}")
    with rasterio.open(args.output_tif, "w", **opt_meta) as dst:
        for b in range(3):
            dst.write(pred_scaled[b], b + 1)
            
    # 6. Save visual PNG false color preview
    if args.output_png:
        save_rgb_false_color(pred_tensor.cpu().squeeze(0), args.output_png)
        
    print("Inference complete!")

if __name__ == "__main__":
    run_inference()
