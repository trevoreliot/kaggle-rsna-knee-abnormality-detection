import pandas as pd
import pydicom
import zarr
import numpy as np
from pathlib import Path
from dagster import asset, Definitions, AssetExecutionContext

DATA_DIR = Path("assets/big-data/rsna-knee-abnormality-detection")
LABELS_DIR = Path("assets/big-data/rsna-knee-abnormality/custom_labels")
OUTPUT_DIR = Path("assets/processed")

@asset
def train_metadata() -> pd.DataFrame:
    """Joins train.csv and train_series.csv on StudyInstanceUID."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    series_df = pd.read_csv(DATA_DIR / "train_series.csv")
    merged_df = pd.merge(train_df, series_df, on="StudyInstanceUID", how="left")
    return merged_df

@asset
def community_labels() -> pd.DataFrame:
    """Loads the community LLM weak-supervision dataset."""
    # Assuming there's a CSV in the LABELS_DIR
    labels_file = LABELS_DIR / "llm_labels.csv"
    if not labels_file.exists():
        # Return empty DF with expected columns if not found to avoid crash during prototyping
        return pd.DataFrame(columns=["StudyInstanceUID", "llm_label"])
    return pd.read_csv(labels_file)

@asset
def merged_metadata(train_metadata: pd.DataFrame, community_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Merges metadata with community labels, establishes label priority, 
    and creates sample_weight.
    """
    if community_labels.empty:
        # Fallback if community labels are not present
        train_metadata["sample_weight"] = 1.0
        return train_metadata
        
    merged = pd.merge(train_metadata, community_labels, on="StudyInstanceUID", how="left")
    
    # Establish priority and sample weight
    # Example logic: assuming 'has_manual_label' or just checking if Kaggle labels are not null
    # Kaggle provides 12 abnormality labels. We check if they are null.
    # For prototype, we'll assume a 'source' column or just check if manual labels are present.
    # Here we simplify: if it was in train.csv natively, it has manual labels.
    # (In Kaggle, train.csv has the manual labels)
    
    merged["sample_weight"] = np.where(merged["llm_label"].notnull() & merged["StudyInstanceUID"].isin(train_metadata["StudyInstanceUID"]), 1.0, 0.85)
    # Refine weight logic based on actual columns in future
    
    return merged

@asset
def zarr_dataset(context: AssetExecutionContext, merged_metadata: pd.DataFrame) -> str:
    """
    Converts DICOM series to chunked Zarr format for fast sequential reads.
    Returns the path to the Zarr group.
    """
    zarr_path = OUTPUT_DIR / "train_images.zarr"
    # Create Zarr group
    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=True)
    
    series_paths = list((DATA_DIR / "train_series").glob("*/*"))
    context.log.info(f"Found {len(series_paths)} series to process.")
    
    # Prototyping: Process a small subset to save time
    for series_path in series_paths[:5]:
        study_id = series_path.parent.name
        series_id = series_path.name
        
        dcm_files = sorted(list(series_path.glob("*.dcm")))
        if not dcm_files:
            continue
            
        slices = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f, force=True)
                a = ds.pixel_array.astype(np.float32)
                sl = float(getattr(ds, 'RescaleSlope', 1) or 1)
                ic = float(getattr(ds, 'RescaleIntercept', 0) or 0)
                a = a * sl + ic
                slices.append(a)
            except Exception as e:
                context.log.warning(f"Error reading {f}: {e}")
                
        if slices:
            vol = np.stack(slices)
            # Create dataset for this series in Zarr
            # Chunking along spatial dimensions (e.g., slices, 256, 256)
            ds_name = f"{study_id}/{series_id}"
            root.create_dataset(
                ds_name,
                data=vol,
                chunks=(1, vol.shape[1], vol.shape[2]),
                dtype=vol.dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
            )
            
    return str(zarr_path)

defs = Definitions(
    assets=[train_metadata, community_labels, merged_metadata, zarr_dataset]
)
