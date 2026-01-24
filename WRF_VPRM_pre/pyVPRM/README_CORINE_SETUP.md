# pyVPRM CORINE Data Processing Setup

This README explains the necessary setup steps and processing workflow for working with CORINE data in pyVPRM for generating VPRM input files for WRF.

## Prerequisites Installation

1. **Install pyVPRM**
    https://github.com/tglauch/pyVPRM

2. **Clone pyVPRM_examples**
   ```bash
   git clone https://github.com/tglauch/pyVPRM_examples.git
   ```

## Critical Modification: CORINE Data Handling

### Important: Modify VPRM.py

Before running any processing jobs, you **must** modify the `VPRM.py` file in your installed pyVPRM package to properly handle CORINE data.

The key modifications are in the `add_land_cover_map()` method. You need to ensure that CORINE data is read and mapped to VPRM classes exactly as shown in `VPRM_modified.py` in this directory.

#### Modification Details

Copy the CORINE data mapping and processing logic from **`VPRM_modified.py`** (specifically the section marked between `# cmr begin` and `# cmr end`) into your installed `VPRM.py`.

The critical section includes:
- Sorted iteration through VPRM class mappings
- Proper XArray `where()` operations for data replacement
- Validation of unique values after replacements

**Example location in VPRM.py to modify:**
```python
def add_land_cover_map(self, land_cover_map, var_name="band_1", ...):
    # ... other code ...
    
    # INSERT THE MODIFICATIONS HERE (from VPRM_modified.py lines 597-611)
    sorted_keys = sorted(self.map_to_vprm_class.keys())
    for key in sorted_keys:
        land_cover_map.sat_img[var_name] = xr.where(
            land_cover_map.sat_img[var_name] == key,
            self.map_to_vprm_class[key],
            land_cover_map.sat_img[var_name],
        )
    
    # Check the unique values after replacements
    print(
        "Unique values after replacements:",
        np.unique(land_cover_map.sat_img[var_name]),
    )
```

## Processing Workflow

Once pyVPRM is properly configured with the CORINE modifications, follow this workflow:

### Step 1: download satellite data

**pyVPRM Documentation**: https://github.com/tglauch/pyVPRM


These jobs will process the satellite data in chunks and generate intermediate output files.

### Step 2: Join Chunks

After all chunk processing jobs complete successfully, use the chunk joining script:

```bash
python join_chunks.py
```

### Step 3: Rename Output Files

Rename the output files to match WRF naming conventions:

```bash
bash rename_output.sh
```

### Step 4: Copy to WRF Directory

Finally, copy the processed VPRM files to the WRF input directory:

```bash
bash copy_vprm_files.sh
```
