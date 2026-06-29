import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, square
from pathlib import Path

class RemoteSensingPreprocessor:
    """
    Handles preprocessing steps for optical and radar inputs:
    1. Co-registering Sentinel-1 SAR bands (VV, VH) to LISS-4 geometry (5.8m GSD).
    2. Generating cloud masks using Normalized Snow-Cloud Index (NSCI) with Otsu thresholding.
    3. Exporting processed inputs to NumPy tensors or GeoTIFFs.
    """
    
    @staticmethod
    def warp_sar_to_optical(optical_path, sar_path, out_bands=2):
        """
        Reprojects and resamples Sentinel-1 SAR bands to match the exact coordinate system,
        transform matrix, and dimensions of a LISS-4 scene.
        """
        with rasterio.open(optical_path) as opt_src:
            opt_transform = opt_src.transform
            opt_crs = opt_src.crs
            H, W = opt_src.height, opt_src.width
            
        sar_warped = np.zeros((out_bands, H, W), dtype=np.float32)
        
        with rasterio.open(sar_path) as sar_src:
            for b in range(min(out_bands, sar_src.count)):
                reproject(
                    source=rasterio.band(sar_src, b + 1),
                    destination=sar_warped[b],
                    src_transform=sar_src.transform,
                    src_crs=sar_src.crs,
                    dst_transform=opt_transform,
                    dst_crs=opt_crs,
                    resampling=Resampling.bilinear
                )
        return sar_warped

    @staticmethod
    def generate_nsci_mask(optical_path, dilation_kernel_size=5):
        """
        Computes the Normalized Snow-Cloud Index (NSCI) for LISS-4 Green and NIR bands:
        NSCI = (Green - NIR) / (Green + NIR)
        Applies Otsu binarization and morphological dilation for shadow capture.
        """
        with rasterio.open(optical_path) as src:
            # LISS-4 band mapping: Band 1 (Green), Band 3 (NIR)
            # Standard band indexes: B1=Green, B2=Red, B3=NIR (1-indexed)
            green = src.read(1).astype(np.float32)
            nir = src.read(3).astype(np.float32)
            
        # Avoid division by zero
        nsci = (green - nir) / (green + nir + 1e-8)
        
        # Determine optimal threshold using Otsu's method
        try:
            thresh = threshold_otsu(nsci)
            mask = (nsci > thresh).astype(np.uint8)
        except ValueError:
            # Fallback if the image lacks bimodal distribution (e.g. completely clear/cloudy)
            mask = (nsci > 0.0).astype(np.uint8)
            
        # Dilate to capture fuzzy cloud boundaries and adjoining shadows
        if dilation_kernel_size > 0:
            mask = dilation(mask, square(dilation_kernel_size))
            
        return mask, nsci

    def preprocess_pair(self, optical_path, sar_path, out_dir):
        """
        Runs the full preprocessing pipeline on an optical-SAR pair and saves results.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        scene_id = Path(optical_path).stem.split('_')[0]
        
        print(f"⚙️ Preprocessing pair: {scene_id}")
        
        # Load optical
        with rasterio.open(optical_path) as src:
            optical_data = src.read().astype(np.float32)
            
        # Warp SAR
        sar_warped = self.warp_sar_to_optical(optical_path, sar_path)
        
        # Generate cloud mask
        cloud_mask, _ = self.generate_nsci_mask(optical_path)
        cloud_frac = cloud_mask.mean() * 100
        
        # Save array files
        np.save(out_dir / f"{scene_id}_liss4.npy", optical_data)
        np.save(out_dir / f"{scene_id}_sar.npy", sar_warped)
        np.save(out_dir / f"{scene_id}_cloudmask.npy", cloud_mask)
        
        print(f"  Saved preprocessed files for {scene_id} (Cloud cover: {cloud_frac:.2f}%)")
        return out_dir / f"{scene_id}_liss4.npy", out_dir / f"{scene_id}_sar.npy", out_dir / f"{scene_id}_cloudmask.npy"

if __name__ == "__main__":
    preprocessor = RemoteSensingPreprocessor()
    print("Preprocessor ready. Call RemoteSensingPreprocessor().preprocess_pair(opt, sar, out)")
