# DoseRAD Preprocessing Pipeline

A preprocessing pipeline for medical imaging data from the SynthRAD2025 dataset, performing deformable image registration and automatic segmentation for radiation therapy planning.

## Overview

This pipeline processes CT, MR, and CBCT medical images to:
- Perform deformable image registration between CT and MR/CBCT scans
- Generate automatic segmentations of anatomical structures
- Prepare data for dose calculation in radiation therapy

## Features

- **Deformable Image Registration**: Uses Convex Adam algorithm with MIND (Modality Independent Neighbourhood Descriptor) features
- **Automatic Segmentation**: Leverages TotalSegmentator for organ-at-risk (OAR) structure segmentation
- **Multi-modality Support**: Handles MR-to-CT and CBCT-to-CT registration
- **Batch Processing**: Configuration-based processing of multiple patients
- **Quality Control**: Generates visualization overlays for registration verification

## Project Structure

```
.
├── preprocessor.py              # Main preprocessing class
├── run_preprocessing.py         # Entry point for batch processing
├── README.md
├── configs/
│   ├── MR_CT_TH_config.yml     # MR thorax dataset configuration
│   ├── CBCT_CT_TH_config.yml   # CBCT thorax dataset configuration
│   ├── CBCT_CT_HN_config.yml   # CBCT head & neck dataset configuration
│   └── generate_config.py       # Utility to generate config files
├── utils/
│   ├── config.py               # Configuration loading utilities
│   ├── io.py                   # Image I/O operations
│   ├── img.py                  # Image processing utilities
│   ├── reg.py                  # Registration algorithms
│   ├── seg.py                  # Segmentation functions
│   ├── vis.py                  # Visualization utilities
│   └── convex_adam_MIND_gaussian.py  # Custom registration implementation
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt
```

## Requirements

- Python 3.13
- CUDA-capable GPU (recommended)
- Docker and Docker Compose

### Python Dependencies

```
numpy
matplotlib
scipy
SimpleITK
convexAdam
TotalSegmentator
ipykernel
pydantic
```

## Installation

### Using Docker (Recommended)

1. Build and run the container:
```bash
cd docker
docker-compose up -d
```

2. Access the container:
```bash
docker-compose exec doserad bash
```

### Manual Installation

```bash
pip install -r docker/requirements.txt
```

## Usage

### Configuration

Patient configurations are stored in YAML files in the `configs/` directory. Each patient entry contains:

```yaml
PATIENT_ID:
  modality: MR|CBCT
  region: TH|HN|AB
  ct_path: /path/to/ct.mha
  input_path: /path/to/mr.mha  # or cbct.mha
  sr_mask: /path/to/mask.mha
  sr_structures: /path/to/structures/
  output_dir: /path/to/output/
  ts_segmentation: true
```

Generate new configurations using:
```python
python configs/generate_config.py
```

### Running the Pipeline

Process all patients in a configuration file:

```python
python run_preprocessing.py
```

Or process individual patients:

```python
from preprocessor import PreProcessor
from utils.config import load_patient_configs

configs = load_patient_configs('./configs/MR_CT_TH_config.yml')
processor = PreProcessor('1THA001', configs['1THA001'])
processor.run_preprocessing()
```

### Pipeline Steps

The preprocessing pipeline executes the following steps for each patient:

## Output Files


## Data Sources

This pipeline processes data from:
- **SynthRAD2025**: Training dataset for Tasks 1 (MR-CT) and 2 (CBCT-CT) (Thummerer et al. Med Phys. 2025; 52:e17981. https://doi.org/10.1002/mp.17981)


## References

- Convex Adam Registration: https://pubmed.ncbi.nlm.nih.gov/39283782/
- TotalSegmentator: https://github.com/wasserth/TotalSegmentator

## License

[Add your license information here]

## Contact

[Add contact information here]
