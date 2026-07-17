"""Inspect spatial profiles of all data types to understand alignment requirements."""
import rasterio
import numpy as np

files = {
    "Sentinel-2 Source B04 (10m ref anchor)": r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon\Pilot_Area_Labels\source_labels\ref_agrifieldnet_competition_v1_source_02160\ref_agrifieldnet_competition_v1_source_02160_B04_10m.tif",
    "Train Label (crop-type)": r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon\Pilot_Area_Labels\train_labels\ref_agrifieldnet_competition_v1_labels_train_02160.tif",
    "Train Label (field_ids)": r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon\Pilot_Area_Labels\train_labels\ref_agrifieldnet_competition_v1_labels_train_02160_field_ids.tif",
    "AWiFS BAND2 (25mar)": r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon\Satellite_data\base_awifs_folder\awifs_25mar2026\BAND2.tif",
}

for name, path in files.items():
    print(f"=== {name} ===")
    try:
        with rasterio.open(path) as src:
            print(f"  Shape (H x W): {src.height} x {src.width}")
            print(f"  Bands:         {src.count}")
            print(f"  Dtype:         {src.dtypes}")
            print(f"  CRS:           {src.crs}")
            print(f"  Resolution:    {src.res}")
            print(f"  Transform:     {src.transform}")
            print(f"  Bounds:        {src.bounds}")
            data = src.read(1)
            print(f"  Data range:    min={np.nanmin(data)}, max={np.nanmax(data)}")
            print(f"  Unique vals:   {len(np.unique(data))} unique values")
            if len(np.unique(data)) < 20:
                print(f"  Unique:        {np.unique(data)}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
