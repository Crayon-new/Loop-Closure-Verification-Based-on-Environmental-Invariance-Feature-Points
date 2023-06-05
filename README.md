# Loop-Closure-Verification-Based-on-Environmental-Invariance-Feature-Points

## Overview

This is the repository for the project of course EEE5346 Autonomous Robot Navigation in SUSTech 2023-Spring. The
description of the course project is in the file [project-description.pdf](doc/project-description.pdf), and the Github
repository is [MedlarTea/EE5346_2023_project](https://github.com/MedlarTea/EE5346_2023_project).

## Environment Setup

## Experiment on Validation Split

### Prepare Data

Clone the repository [MedlarTea/EE5346_2023_project](https://github.com/MedlarTea/EE5346_2023_project) and unzip the data.

```bash
git clone https://github.com/MedlarTea/EE5346_2023_project
cd EE5346_2023_project
unzip -q '*.zip'
```

### Prepare Background Mask

**Method 1: Use the prepared data**

Build the data symlinks and unzip the mask files

```bash
ln -s /path/to/repository/Loop-Closure-Verification-Based-on-Environmental-Invariance-Feature-Points/data/*_mask.zip /path/to/repository/EE5346_2023_project
cd /path/to/repository/EE5346_2023_project
unzip -q '*_mask.zip'
```

**Method 2: Use the pretrained model**

Download the checkpoint file [from models](https://www.dropbox.com/s/fmlq806p2wqf311/trained_models.zip?dl=0) and unzip
it to DANNet/checkpoint

```bash
cd Loop-Closure-Verification-Based-on-Environmental-Invariance-Feature-Points/DANNet
# compute mask, input "Autumn_mini_query" can be "Autumn_mini_query", "Night_mini_ref" or "Suncloud_mini_ref", the output mask will be saved in dir like "Autumn_mini_query_mask"
python evaluate.py --input /path/to/repository/EE5346_2023_project/Autumn_mini_query
```

### Run Validation

Run python script [lcv_validation.py](lcv_validation.py)

```bash
cd /path/to/repository/Loop-Closure-Verification-Based-on-Environmental-Invariance-Feature-Points
python lcv_validation.py --data_root_dir /path/to/repository/EE5346_2023_project --save_dir ./output
```

## Experiment on Test Split

### Prepare Data

Following [MedlarTea/EE5346_2023_project](https://github.com/MedlarTea/EE5346_2023_project#final-testing)

### Prepare Background Mask

Following the steps in [Experiment on Validation Split](#experiment-on-validation-split)

### Run Test

```bash
python lcv_test.py --test_file /path/to/test_file --data_root_dir /path/to/data/for/test --save_dir ./output_for_test
```

example for test_file and data dir for test:

test_file.txt

```text
scene_1/000001.png scene_2/000001.png
scene_1/000002.png scene_2/000003.png
```

data for test directory structure

```text
data_for_test
├── scene_1
│      ├── 000001.png
│      ├── 000002.png
│      └── 000003.png
├──scene_1_mask
│      ├── 000001.npy
│      ├── 000002.npy
│      └── 000003.npy
├──scene_2
│      ├── 000001.png
│      ├── 000002.png
│      └── 000003.png
└──scene_2_mask
       ├── 000001.npy
       ├── 000002.npy
       └── 000003.npy

```

## Acknowledgement

* [ml-aspanformer](https://github.com/apple/ml-aspanformer)
* [DANNet](https://github.com/W-zx-Y/DANNet)