# Loop-Closure-Verification-Based-on-Environmental-Invariance-Feature-Points

## Overview

This is the repository for the project of course EEE5346 Autonomous Robot Navigation in SUSTech 2023-Spring. The
description of the course project is in the file [project-description.pdf](doc/project-description.pdf), and the Github
repository is [MedlarTea/EE5346_2023_project](https://github.com/MedlarTea/EE5346_2023_project).

## Environment Setup

## Experiment on Validation Split

### Prepare Data

Clone this [repository](https://github.com/MedlarTea/EE5346_2023_project)

```bash
git clone https://github.com/MedlarTea/EE5346_2023_project
```

Build the dataset symlinks

```bash
ln -s /path/to/repository/EE5346_2023_project /path/to/repository/Loop-Closure-Verification-Based-on-Environmental-Invariance-Feature-Points/data
```

### Prepare Background Mask

### Run Validation

Run python script [lcv_validation.py](lcv_validation.py)

```bash
cd /path/to/repository/Loop-Closure-Verification-Based-on-Environmental-Invariance-Feature-Points
python lcv_validation.py --data_root_dir ./data --save_dir ./output
```

## Experiment on Test Split

### Prepare Data

### Prepare Background Mask

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
└── scene_2
       ├── 000001.png
       ├── 000002.png
       └── 000003.png
```

## Acknowledgement

https://github.com/apple/ml-aspanformer