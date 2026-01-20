# WRF/VPRM Setup Guide

## Step 0: Directory Structure

**Home folder:**
Clone the repos and replace all path names with your own. 
```
clone https://github.com/Matthias-Reif-PhD/WRF_VPRM_inComplexTopo
```


```
Github/
├── plotting_codes
├── VPRM_tools
├── WRF_VPRM_post
└── WRF_VPRM_pre
```

find all old path and replace them with yours

```
grep -ir /home/c707/c7071034
grep -ir /scratch/c7071034
```


**Scratch folder:**
Clone the WRF and WPS repos 
```
clone https://github.com/Matthias-Reif-PhD/WRF
clone https://github.com/Matthias-Reif-PhD/WPS
```

```
/scratch/
├── DATA/
│   ├── CAMS/
│   ├── CORINE_LC/
│   ├── ECMWF/
│   ├── MODIS/
│   ├── pyVPRM/
│   ├── vprm_shapeshifter/
│   ├── WPS_GEOG/
│   └── VPRM_input/
├── WPS/
├── WRF_54km/
├── WRF_27km/
├── WRF_9km/
└── WRF_3km/
```

Install WRF/WPS: https://www2.mmm.ucar.edu/wrf/OnLineTutorial/

## Step 1: CAMS Data (CO₂)

Setup API credentials (`$HOME/.cdsapirc`):

```
url: https://ads.atmosphere.copernicus.eu/api
key: YOUR_KEY
```

Download and merge:

```bash
cd $HOME/Github/WRF_VPRM_pre/CAMS
bash job_get_CAMS_CO2.sh
bash job_get_CAMS_lnsp.sh
bash job_get_CAMS_surface.sh
bash merge_CAMS_month_to_year.sh
```

## Step 2: ERA5 Data

Change CDS API URL in `$HOME/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
```

Download:
```bash
cd /scratch/DATA/ECMWF/pressure && python get_ECMWF_pressure.py
cd /scratch/DATA/ECMWF/surface && python get_ECMWF_surface.py
```

## Step 3: CORINE Land Cover

```bash
cd $HOME/Github/WRF_VPRM_pre/CORINE
./reproject_Corine_for_WRF.sh
./reproject_Corine_for_VPRM.sh
```

## Step 4: VPRM Inputs

1. Clone & setup (see "Installation" in README) [pyVPRM](https://github.com/tglauch/pyVPRM)
2. Download satellite data (MODIS)
3. Run pyVPRM preprocessing
4. Clone & run [vprm_shapeshifter](https://github.com/michal-galkowski/vprm_shapeshifter)
5. Copy outputs to `/scratch/DATA/VPRM_input/`

## Step 5: Boundary Conditions

Prepare `ecmwf_coeffs_L60.csv` and `interp_indices.txt.npz` for your model levels.

## Next: Run Simulations

See `run_WPS_WRF/README_run_WPS_WRF.md`

