import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import rasterio

class SEN12MS_LISS4_SimulationDataset(Dataset):
    """
    Simulates LISS-4 imagery using the 13-band Sentinel-2 data from SEN12MS-CR.
    Sentinel-2 bands used:
      - B3 (Green) -> index 2
      - B4 (Red) -> index 3
      - B8 (NIR) -> index 7
    """
    # 0-indexed Sentinel-2 bands matching LISS-4: Green, Red, NIR
    LISS4_BAND_MAP = [2, 3, 7]

    def __init__(self, root_dir, split='train', patch_size=256):
        self.root = Path(root_dir)
        self.patch_size = patch_size
        self.pairs = []
        
        # Traverse all directories in the split
        if self.root.exists():
            for season_dir in self.root.iterdir():
                if not season_dir.is_dir():
                    continue
                s1_dir = season_dir / 'ROIs1158_spring_s1'
                s2c_dir = season_dir / 'ROIs1158_spring_s2_cloudy'
                s2cf_dir = season_dir / 'ROIs1158_spring_s2'
                
                # Fallback support for standard folder names
                if not s1_dir.exists():
                    s1_dir = season_dir / 's1'
                if not s2c_dir.exists():
                    s2c_dir = season_dir / 's2_cloudy'
                if not s2cf_dir.exists():
                    s2cf_dir = season_dir / 's2_cloud_free'
                
                if s1_dir.exists() and s2c_dir.exists() and s2cf_dir.exists():
                    s1_files = sorted(s1_dir.glob('**/*.tif'))
                    for s1_f in s1_files:
                        filename = s1_f.name
                        if '_s1_' in filename:
                            # Extract suffix index (e.g. "1_p15" from "ROIs1158_spring_s1_1_p15.tif")
                            idx = filename.split('_s1_')[-1].replace('.tif', '')
                            s2c_f = s2c_dir / f"ROIs1158_spring_s2_cloudy_{idx}.tif"
                            s2cf_f = s2cf_dir / f"ROIs1158_spring_s2_{idx}.tif"
                            
                            # Fallback if standard suffix is used
                            if not s2c_f.exists():
                                s2c_f = s2c_dir / f"s2_{idx}.tif"
                            if not s2cf_f.exists():
                                s2cf_f = s2cf_dir / f"s2_{idx}.tif"
                                
                            if s2c_f.exists() and s2cf_f.exists():
                                self.pairs.append((s1_f, s2c_f, s2cf_f))
        
        n_samples = len(self.pairs)
        if split == 'train':
            self.pairs = self.pairs[:int(0.85 * n_samples)]
        else:
            self.pairs = self.pairs[int(0.85 * n_samples):]
            
        print(f"📁 Loaded SEN12MS-CR dataset split '{split}' with {len(self.pairs)} sample pairs.")

    def __len__(self):
        return len(self.pairs)

    def _read_tif(self, path, bands=None):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        if bands is not None:
            data = data[bands]
        return data

    def __getitem__(self, idx):
        s1_path, s2c_path, s2cf_path = self.pairs[idx]
        
        # Read files
        sar = self._read_tif(s1_path)                         # (2, H, W)
        cloudy_s2 = self._read_tif(s2c_path)                 # (13, H, W)
        clear_s2 = self._read_tif(s2cf_path)                 # (13, H, W)
        
        # Extract LISS-4 bands
        cloudy = cloudy_s2[self.LISS4_BAND_MAP]              # (3, H, W)
        clear = clear_s2[self.LISS4_BAND_MAP]                # (3, H, W)
        
        # Normalize
        # SAR is backscatter in dB, usually normalized to [-1, 1] by dividing by 10000.0 or clipping
        sar = np.clip(sar / 10000.0, -1.0, 1.0)
        # Optical is DN value, normalize to [0, 1]
        cloudy = np.clip(cloudy / 3000.0, 0.0, 1.0)
        clear = np.clip(clear / 3000.0, 0.0, 1.0)
        
        # Random crop to patch_size
        H, W = cloudy.shape[1], cloudy.shape[2]
        if H > self.patch_size and W > self.patch_size:
            top = np.random.randint(0, H - self.patch_size)
            left = np.random.randint(0, W - self.patch_size)
            slice_obj = (slice(None), slice(top, top + self.patch_size), slice(left, left + self.patch_size))
            sar = sar[slice_obj]
            cloudy = cloudy[slice_obj]
            clear = clear[slice_obj]
            
        # Data Augmentation (Flips)
        if np.random.rand() > 0.5:
            sar = np.flip(sar, axis=1).copy()
            cloudy = np.flip(cloudy, axis=1).copy()
            clear = np.flip(clear, axis=1).copy()
        if np.random.rand() > 0.5:
            sar = np.flip(sar, axis=2).copy()
            cloudy = np.flip(cloudy, axis=2).copy()
            clear = np.flip(clear, axis=2).copy()
            
        return (
            torch.tensor(sar, dtype=torch.float32),
            torch.tensor(cloudy, dtype=torch.float32),
            torch.tensor(clear, dtype=torch.float32)
        )

class LISS4RealDataset(Dataset):
    """
    Dataset loader for actual preprocessed LISS-4 and Sentinel-1 SAR pairs.
    Reads pre-saved numpy files from the preprocessing directory.
    """
    def __init__(self, processed_dir, patch_size=256):
        self.dir = Path(processed_dir)
        self.patch_size = patch_size
        
        # Find all unique scene prefixes
        self.scenes = sorted(set(
            f.name.replace('_liss4.npy', '').replace('_sar.npy', '').replace('_cloudmask.npy', '')
            for f in self.dir.glob('*.npy')
        ))
        
        print(f"📁 Loaded real LISS-4 dataset with {len(self.scenes)} scene pairs.")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        sid = self.scenes[idx]
        
        # Load numpy arrays
        liss4 = np.load(self.dir / f"{sid}_liss4.npy")        # (3, H, W)
        sar = np.load(self.dir / f"{sid}_sar.npy")            # (2, H, W)
        mask = np.load(self.dir / f"{sid}_cloudmask.npy")      # (H, W)
        
        # Normalization
        liss4 = np.clip(liss4 / 10000.0, 0.0, 1.0)
        sar = np.clip(sar / 1000.0, -1.0, 1.0)
        mask = mask.astype(np.float32)
        
        # Random crop to patch_size
        H, W = liss4.shape[1], liss4.shape[2]
        if H > self.patch_size and W > self.patch_size:
            top = np.random.randint(0, H - self.patch_size)
            left = np.random.randint(0, W - self.patch_size)
            slice_3d = (slice(None), slice(top, top + self.patch_size), slice(left, left + self.patch_size))
            slice_2d = (slice(top, top + self.patch_size), slice(left, left + self.patch_size))
            
            liss4 = liss4[slice_3d]
            sar = sar[slice_3d]
            mask = mask[slice_2d]
            
        return (
            torch.tensor(sar, dtype=torch.float32),
            torch.tensor(liss4, dtype=torch.float32),  # Cloudy optical input
            torch.tensor(mask, dtype=torch.float32)    # Cloud mask
        )
