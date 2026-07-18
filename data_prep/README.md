# Dataset construction (one-time)

Scripts that built the training corpus (`Extracted_dataset_gee/chips/`,
1,151 chips) from the raw AgriFieldNet tiles and GEE exports. Needed only
to reproduce the dataset from scratch — the runtime pipeline never calls
them. Run order: `extract_dataset.py` → `extract_combined.py` →
`extract_gee.py`.
