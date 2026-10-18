import os
from pathlib import Path
import pydicom
import numpy as np
import cv2
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count

DATA_DIR = Path("assets/big-data/rsna-knee-abnormality-detection")
TRAIN_SERIES_DIR = DATA_DIR / "train_series"
OUTPUT_DIR = Path("assets/processed/2d_slices/train")

def process_series(series_path: Path):
    """
    Reads all DICOMs in a series, applies volumetric normalization, 
    and saves them as PNGs.
    """
    study_id = series_path.parent.name
    series_id = series_path.name
    out_series_dir = OUTPUT_DIR / study_id / series_id
    
    if out_series_dir.exists() and len(list(out_series_dir.glob("*.png"))) > 0:
        return # Already processed
        
    out_series_dir.mkdir(parents=True, exist_ok=True)
    
    dcm_files = sorted(list(series_path.glob("*.dcm")))
    if not dcm_files:
        return
        
    slices = []
    metadata = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, 'RescaleSlope', 1) or 1)
            ic = float(getattr(ds, 'RescaleIntercept', 0) or 0)
            a = a * sl + ic
            slices.append(a)
            metadata.append(f.name)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not slices:
        return
        
    vol = np.stack(slices)
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)
    vol = (vol * 255).astype(np.uint8)
    
    for i, (img, name) in enumerate(zip(vol, metadata)):
        png_name = name.replace(".dcm", ".png")
        out_path = out_series_dir / png_name
        cv2.imwrite(str(out_path), img)

def main():
    series_paths = []
    for study in TRAIN_SERIES_DIR.iterdir():
        if study.is_dir():
            for series in study.iterdir():
                if series.is_dir():
                    series_paths.append(series)
                    
    print(f"Found {len(series_paths)} series to process.")
    
    # We use process_map for multiprocessing with tqdm
    process_map(process_series, series_paths, max_workers=cpu_count(), chunksize=10)
    print("Done!")

if __name__ == "__main__":
    main()
