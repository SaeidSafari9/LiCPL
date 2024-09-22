
# LiCS-PL

LiCS-PL is an open-source Python package for InSAR time series analysis using **wrapped interferograms** from LiCSAR products. Unlike LiCSBAS, which uses unwrapped interferograms, LiCS-PL processes wrapped interferograms to generate a consistent single-master wrapped interferogram stack using two Phase Linking (PL) methods:

- **Eigenvalue Decomposition (EVD)** ([doi:10.1109/TGRS.2014.2352853](https://doi.org/10.1109/TGRS.2014.2352853))
- **Eigendecomposition-based Maximum-likelihood estimator of Interferometric phase (EMI)** ([doi:10.1109/TGRS.2018.2826045](https://doi.org/10.1109/TGRS.2018.2826045))

## Features

- Processes wrapped interferograms and coherence data from LiCSAR products.
- Implements EVD and EMI phase linking methods.
- Generates Temporal Coherence maps to evaluate the quality of linked phases.
- Recreates interferogram networks based on user-defined configurations.
- Compatible with LiCSBAS for unwrapping and further time series analysis.

## Preparation

### 1. Download LiCSAR Products

- Visit the [COMET-LiCS web portal](https://comet.nerc.ac.uk/COMET-LiCS-portal/).
- Select the desired frame ID and download the geocoded GeoTIFF files of:
  - **`diff_pha`** (wrapped interferogram)
  - **`cc`** (coherence)

### 2. Organize Data

- Create a directory structure identical to the `GEOC` folder used by LiCSBAS during Step 0-1.
  - Each GeoTIFF file should be placed in a folder named after its interferogram date pair.
- Example structure:

  ```
  root_path/
  └── GEOC/
      ├── 20180101_20180113/
      │   ├── 20180101_20180113.geo.diff_pha.tif
      │   └── 20180101_20180113.geo.cc.tif
      ├── 20180113_20180125/
      │   ├── 20180113_20180125.geo.diff_pha.tif
      │   └── 20180113_20180125.geo.cc.tif
      └── ...
  ```

### 3. Configure the Project

- Edit the `config.txt` file:
  - Set `root_path` to the path where the `GEOC` folder is located.
  - Adjust `network_config` and other parameters as needed.

#### Additional Configurations:

- **`GoF`**: Goodness of fit or temporal coherence (based on [doi:10.1109/TGRS.2011.2124465](https://doi.org/10.1109/TGRS.2011.2124465)). Default is `0.4`, and it can be set between `0` and `1`. This value is used to remove PL results with low temporal coherence.
- **`avg_coh`**: Average coherence of the interferograms stack. Since LiCSAR products use coherence values between `0` and `255`, this is the valid range for this setting. The default is `10`.
- **`patches_nRows`**: The number of rows for each patch created based on the original stack. This value can be decreased for systems with low RAM. The default is `10`.

### 4. Install Dependencies

- Install the required Python packages:

## Requirements

- **Python 3.6 and above**
- **Python packages**:
  - joblib
  - matplotlib
  - numpy
  - gdal
  - scipy

## Usage

Run the main processing script:

```bash
python PL_batchProcessing.py
```

### Processing Steps

#### **Step 00: Data Preparation**

- Reads all interferograms from the `GEOC` folder.
- Creates Sample Correlation Matrices (SCM) from the interferograms.

#### **Step 01: Phase Linking Methods**

- Performs the EVD and EMI phase linking methods on the SCM matrices.
- Calculates Temporal Coherence maps from the linked phase results.

#### **Step 02: Interferogram Network Reconstruction**

- Recreates the interferogram network based on the `network_config` in `config.txt`.
  - By default, generates short-term interferograms with temporal baselines ≤18 days.
  - Optionally, can recreate the original interferogram network from the `GEOC` folder.
- **GoF** and **avg_coh** maskings are applied during this step to filter out low-quality results.

### Outputs

- Two new folders will be created alongside `GEOC`:

  ```
  root_path/
  ├── GEOC/
  ├── EVD/
  │   └── GEOC/
  └── EMI/
      └── GEOC/
  ```

- `EVD/GEOC/` and `EMI/GEOC/` contain the processed data, ready for unwrapping and time series analysis using LiCSBAS.

## Configuration

Adjust settings in `config.txt`:

- **`root_path`**: Path to the directory containing the `GEOC` folder.
- **`network_config`**: Defines the interferogram network to be recreated (`short_term`, `original`, etc.).
- **`GoF`**, **`avg_coh`**, and **`patches_nRows`** as explained above.

## Integration with LiCSBAS

- The outputs from LiCS-PL are compatible with LiCSBAS.
- Proceed with LiCSBAS processing starting from the unwrapping step using the `EVD/GEOC/` and `EMI/GEOC/` folders.

## Example Workflow

1. **Data Preparation**

   - Download and organize LiCSAR products.

2. **Phase Linking with LiCS-PL**

   - Run `PL_batchProcessing.py` to perform phase linking and generate new interferogram stacks.

3. **Time Series Analysis with LiCSBAS**

   - Use LiCSBAS to unwrap and analyze the processed data from LiCS-PL.

---

**Disclaimer**: This is research code provided "as is" with no warranties of correctness. Use at your own risk.

## Acknowledgments

- **[LiCSAR](https://comet.nerc.ac.uk/COMET-LiCS-portal/)**: For providing the Sentinel-1 InSAR products.
- **[LiCSBAS](https://github.com/comet-licsar/LiCSBAS)**: For software structure inspiration and compatibility.
- **Phase Linking Methods References**:
  - EVD: [doi:10.1109/TGRS.2014.2352853](https://doi.org/10.1109/TGRS.2014.2352853)
  - EMI: [doi:10.1109/TGRS.2018.2826045](https://doi.org/10.1109/TGRS.2018.2826045)
