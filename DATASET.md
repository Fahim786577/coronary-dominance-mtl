# Dataset and Split Preparation

This file documents the dataset organization expected by the `coronary-dominance-mtl` repository.

The repository is designed to run on **extracted angiography frames** and **CSV split files**. It does not distribute the raw CoronaryDominance dataset, DICOM files, original `.npz` archives, full manuscript split CSVs, or verified model checkpoints inside the GitHub repository.

The public `splits/` folder may contain small example/smoke-test split files. These files are intended to verify that the code runs and that the path format is correct. They should not be treated as the full paper splits.

---

## 1. Dataset Source

The associated manuscript uses the **CoronaryDominance** dataset introduced by Kruzhilov et al., *Scientific Data*, 2025:

```text
Kruzhilov, I. et al. CoronaryDominance: Angiogram dataset for coronary dominance classification.
Scientific Data 12, 341 (2025). https://doi.org/10.1038/s41597-025-04676-8
```

The dataset contains invasive coronary angiography studies for binary coronary dominance classification. Each study contains multi-view X-ray angiographic videos from the right coronary artery (RCA) and/or left coronary artery (LCA). The official dataset is distributed separately from this repository through the CoronaryDominance dataset page.

Official dataset source:

```text
https://huggingface.co/datasets/BearSubj13/CoronaryDominance
```

This repository does not mirror or redistribute the raw CoronaryDominance archives. Users should download the dataset from the official source and follow the dataset license/access terms. The official dataset is provided as `.npz` angiographic-view files; this repository expects extracted frame files and task-specific split CSVs prepared from those `.npz` files.

The original CoronaryDominance dataset contains:

```text
1,574 angiographic studies
binary dominance labels: left dominance / right dominance
three dataset parts: main, real distribution, domain shift
```

The official dataset is stored as compressed NumPy `.npz` angiographic-view files, not as extracted `.png` or `.jpg` frames. This repository expects the user to first prepare extracted frame folders and task-specific split CSV files.

---

## 2. Official Dataset Structure vs This Repository Structure

There are two different structures to keep clear:

```text
Official CoronaryDominance dataset structure
    Original dataset layout from the dataset authors.
    Dominance folders contain study folders, and each study contains RCA/LCA .npz view files.

Repository-prepared frame structure
    Layout expected by this repository after extracting frames and preparing task-specific labels.
    The training/evaluation scripts read image files listed in CSV split files.
```

The official dataset structure is approximately:

```text
CoronaryDominance/
├── Left_Dominance/
│   ├── Study0xxxx_<study_id>/
│   │   ├── LCA/
│   │   │   ├── <series_id>.npz
│   │   │   └── ...
│   │   └── RCA/
│   │       ├── <series_id>.npz
│   │       └── ...
│   └── ...
└── Right_Dominance/
    ├── Study0xxxx_<study_id>/
    │   ├── LCA/
    │   │   ├── <series_id>.npz
    │   │   └── ...
    │   └── RCA/
    │       ├── <series_id>.npz
    │       └── ...
    └── ...
```

The prepared structure used by this repository is described in Section 6.

---

## 3. Official Dataset Parts and Categories

The official dataset contains three main parts:

| Dataset part | Studies | Left dominance | Right dominance | Imaging system | Suggested role |
| --- | ---: | ---: | ---: | --- | --- |
| Main dataset | 1,025 | 319 | 706 | Philips Allura Clarity | training / validation / internal testing |
| Real distribution | 400 | 54 | 346 | Philips Allura Clarity | real-world distribution testing |
| Domain shift | 149 | 52 | 97 | Philips Azurion | domain-shift testing |

The main dataset is divided into five categories:

| Main dataset category | Studies | Left dominance | Right dominance |
| --- | ---: | ---: | ---: |
| Bad quality / Poor quality | 28 | 13 | 15 |
| Artefacts / Artifacts | 50 | 15 | 35 |
| Uncertainty / RCA small diameter | 25 | 9 | 16 |
| RCA occlusion | 143 | 31 | 112 |
| Normal | 779 | 251 | 528 |
| **Total** | **1,025** | **319** | **706** |

The dataset authors recommend using the Normal and Occlusion categories from the main dataset for model training and reserving other categories, such as Artefacts, Uncertainty/RCA small diameter, Real Distribution, and Domain Shift, for testing. The associated manuscript uses a task-specific preprocessing strategy described later in this file.

---

## 4. Official `.npz` View Files

Each official `.npz` file represents one angiographic view/video. The important field for frame extraction is:

```text
pixel_array
```

The relevant `.npz` fields described in the dataset paper include:

| Field | Description |
| --- | --- |
| `pixel_array` | Angiographic video array with shape `frames × 512 × 512`; pixel values range from 0 to 255 |
| `seriesid` | Unique ID of the angiographic view |
| `studyid` | Unique ID of the angiographic study |
| `series_number` | Study sequence number used for easier referencing |
| `is_collaterals` | Collaterals in LCA |
| `primary_angle` | Positioner primary angle |
| `secondary_angle` | Positioner secondary angle |
| `is_occlusion` | RCA occlusion tag |
| `is_undefined_type` | High-uncertainty / undefined dominance type tag |
| `is_artifact` | Artefact tag |
| `artery_type` | `LCA` or `RCA` |

The repository training scripts do not currently read `.npz` files directly. Convert or extract the frames from `pixel_array` into image files before using the repository scripts.

Recommended frame naming:

```text
frame_000001.png
frame_000002.png
frame_000003.png
...
```

Keep the original frame order inside each angiographic view, especially for occlusion detection, where temporal sampling is used.

---

## 5. Repository Roots

This repository expects two roots:

```text
<DATA_ROOT>
<SPLIT_ROOT>
```

where:

```text
<DATA_ROOT>   contains extracted image/frame files
<SPLIT_ROOT>  contains CSV files defining train/validation/test splits
```

Example:

```text
D:\CoronaryArteryDominance\data
D:\CoronaryArteryDominance\coronary-dominance-mtl\splits
```

or, from inside the repository:

```bash
python scripts/verify_splits.py --data_root <DATA_ROOT> --split_root splits
```

---

## 6. Expected `<DATA_ROOT>` Layout

The recommended portable frame layout is:

```text
<DATA_ROOT>/
├── dominance/
│   ├── DATA_RCA/
│   │   ├── rightdom/
│   │   └── leftdom/
│   └── DATA_LCA/
│       ├── rightdom/
│       └── leftdom/
├── framequality/
│   ├── DATA_RCA/
│   │   ├── informative/
│   │   └── noninformative/
│   └── DATA_LCA/
│       ├── informative/
│       └── noninformative/
└── occlusion/
    └── DATA_RCA/
        ├── occluded/
        └── nonoccluded/
```

Example frame paths:

```text
<DATA_ROOT>/dominance/DATA_RCA/rightdom/Study001/seq_001/frame_000001.png
<DATA_ROOT>/dominance/DATA_LCA/leftdom/Study002/seq_003/frame_000001.png
<DATA_ROOT>/framequality/DATA_RCA/informative/Study003/seq_001/frame_000001.png
<DATA_ROOT>/framequality/DATA_LCA/noninformative/Study004/seq_002/frame_000001.png
<DATA_ROOT>/occlusion/DATA_RCA/occluded/Study005/seq_001/frame_000001.png
```

The `filename` entries in the split CSV files should be consistent with the dataset loader’s path convention. The recommended approach is to store paths relative to the class folder.

For example, this CSV row:

```csv
filename,label,study_id,artery,split,task,fold
Study001/seq_001/frame_000001.png,rightdom,Study001,RCA,train,dominance,1
```

is resolved as:

```text
<DATA_ROOT>/dominance/DATA_RCA/rightdom/Study001/seq_001/frame_000001.png
```

Do not include the task, artery, or label directory twice. If the loader already constructs:

```text
<DATA_ROOT>/<task>/DATA_<artery>/<label>/<filename>
```

then `filename` should not also start with:

```text
dominance/DATA_RCA/rightdom/
```

---

## 7. Supported Tasks and Label Maps

The repository supports three binary classification tasks.

| Task | Internal task name | Artery views | Labels |
| --- | --- | --- | --- |
| Dominance classification | `dominance` | RCA and LCA | `rightdom`, `leftdom` |
| Frame quality assessment | `framequality` | RCA and LCA | `noninformative`, `informative` |
| RCA occlusion detection | `occlusion` | RCA only | `nonoccluded`, `occluded` |

Important naming note:

```text
CSV/folder task name:       framequality
model output dictionary:    frame_quality
```

The label mappings are:

```python
LABEL_MAPS = {
    "occlusion": {
        "nonoccluded": 0,
        "occluded": 1,
    },
    "framequality": {
        "noninformative": 0,
        "informative": 1,
    },
    "dominance": {
        "rightdom": 0,
        "leftdom": 1,
    },
}
```

Official dataset labels should be mapped as follows:

| Official dataset folder | Repository label |
| --- | --- |
| `Left_Dominance` | `leftdom` |
| `Right_Dominance` | `rightdom` |

---

## 8. Expected `<SPLIT_ROOT>` Layout

The split directory should follow this structure:

```text
<SPLIT_ROOT>/
├── dominance/
│   ├── DATA_RCA/
│   │   └── labels/
│   └── DATA_LCA/
│       └── labels/
├── framequality/
│   ├── DATA_RCA/
│   │   └── labels/
│   └── DATA_LCA/
│       └── labels/
└── occlusion/
    └── DATA_RCA/
        └── labels/
```

For the default repository layout, `<SPLIT_ROOT>` is usually:

```text
splits
```

The public repository may include example CSV files under this structure. These are for smoke testing only unless explicitly replaced with the full paper split files.

---

## 9. Split CSV Format

Each split CSV should use the enriched format:

```csv
filename,label,study_id,artery,split,task,fold
```

Column definitions:

| Column | Required | Description |
| --- | --- | --- |
| `filename` | Yes | Frame filename or relative frame path consistent with `<DATA_ROOT>` |
| `label` | Yes | Class label, e.g. `rightdom`, `leftdom`, `informative`, `occluded` |
| `study_id` | Yes | Study-level identifier used to prevent data leakage |
| `artery` | Yes | `RCA` or `LCA` |
| `split` | Yes | `train`, `val`, or `test` |
| `task` | Yes | `dominance`, `framequality`, or `occlusion` |
| `fold` | Yes | Fold number, e.g. `1` |

Example:

```csv
filename,label,study_id,artery,split,task,fold
Study001/seq_001/frame_000001.png,rightdom,Study001,RCA,train,dominance,1
Study001/seq_001/frame_000002.png,rightdom,Study001,RCA,train,dominance,1
Study002/seq_001/frame_000001.png,leftdom,Study002,RCA,val,dominance,1
```

Recommended file naming pattern:

```text
<task>_<split>_labels_fold_<fold>.csv
```

Examples:

```text
dominance_train_labels_fold_1.csv
dominance_val_labels_fold_1.csv
dominance_test_labels_fold_1.csv

framequality_train_labels_fold_1.csv
framequality_val_labels_fold_1.csv
framequality_test_labels_fold_1.csv

occlusion_train_labels_fold_1.csv
occlusion_val_labels_fold_1.csv
occlusion_test_labels_fold_1.csv
```

If your local code or earlier files use a slightly different naming convention, keep the naming consistent with the scripts in your repository and verify with `scripts/verify_splits.py`.

---

## 10. Study-Level Split Requirement

All splitting must be performed at the **study level**.

Frames from the same angiographic study must not appear in more than one split. For example, if `Study001` is assigned to the training split, then every RCA/LCA frame associated with `Study001` must remain in the training split only.

This is important because frame-level random splitting would leak highly similar frames from the same study across train, validation, and test sets, resulting in overestimated performance.

---

## 11. Task-Specific Dataset Preparation

The three tasks use different preprocessing and sampling procedures.

### 11.1 Frame Quality Assessment

Purpose:

```text
Classify individual frames as informative or non-informative for dominance classification.
```

Labels:

```text
informative
noninformative
```

Artery views:

```text
RCA and LCA
```

In the manuscript experiments, frame quality labels were prepared from 56 Normal studies: the first 28 left-dominant and first 28 right-dominant studies. Frames were manually annotated as informative or non-informative, and splits were made at the study level.

Manuscript split summary:

| Split | Studies | LCA informative | LCA non-informative | RCA informative | RCA non-informative |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 35 | 2994 | 2842 | 1229 | 1227 |
| Validation | 9 | 826 | 816 | 272 | 289 |
| Test | 12 | 947 | 952 | 442 | 347 |
| Total | 56 | 4767 | 4610 | 1943 | 1863 |

Recommended folder examples:

```text
<DATA_ROOT>/framequality/DATA_RCA/informative/
<DATA_ROOT>/framequality/DATA_RCA/noninformative/
<DATA_ROOT>/framequality/DATA_LCA/informative/
<DATA_ROOT>/framequality/DATA_LCA/noninformative/
```

---

### 11.2 RCA Occlusion Detection

Purpose:

```text
Classify RCA videos/sequences as occluded or non-occluded.
```

Labels:

```text
occluded
nonoccluded
```

Artery views:

```text
RCA only
```

Occlusion detection is not defined for LCA in this repository.

In the manuscript experiments, all studies from the Occlusions subset were treated as positive cases. One study without RCA video data was excluded, resulting in 142 occluded studies. The non-occluded class was prepared from 143 Normal studies. Only RCA videos were used.

Manuscript split summary:

| Split | Occluded studies | Occluded frames | Non-occluded studies | Non-occluded frames |
| --- | ---: | ---: | ---: | ---: |
| Train | 90 | 7392 | 91 | 8212 |
| Validation | 23 | 1915 | 23 | 1973 |
| Test | 29 | 2638 | 29 | 2510 |
| Total | 142 | 11945 | 143 | 12695 |

Recommended folder examples:

```text
<DATA_ROOT>/occlusion/DATA_RCA/occluded/
<DATA_ROOT>/occlusion/DATA_RCA/nonoccluded/
```

Training note:

The occlusion model uses temporal clips. The default command examples use:

```text
--clip_length 15
```

The loader samples or pads clips to the required sequence length. Keep the frame ordering within each study/sequence consistent so that temporal sampling is meaningful.

---

### 11.3 Dominance Classification

Purpose:

```text
Classify frames/studies as right-dominant or left-dominant.
```

Labels:

```text
rightdom
leftdom
```

Artery views:

```text
RCA and LCA
```

In the manuscript experiments, 586 studies were selected for dominance classification, consisting of 173 left-dominant and 413 right-dominant studies. A separate 50-study holdout set was reserved for final evaluation. Before dominance training, non-informative frames were removed using a frame filtering pipeline combining frame-quality classification and segmentation-based quality assessment.

Manuscript split summary for the 586-study dominance set:

| Split | Left studies | Right studies | LCA left frames | LCA right frames | RCA left frames | RCA right frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 110 | 264 | 14210 | 27385 | 1599 | 11880 |
| Validation | 28 | 66 | 3473 | 7291 | 438 | 2955 |
| Test | 35 | 83 | 4269 | 8552 | 468 | 3681 |
| Total | 173 | 413 | 21952 | 43228 | 2505 | 18516 |

Recommended folder examples:

```text
<DATA_ROOT>/dominance/DATA_RCA/rightdom/
<DATA_ROOT>/dominance/DATA_RCA/leftdom/
<DATA_ROOT>/dominance/DATA_LCA/rightdom/
<DATA_ROOT>/dominance/DATA_LCA/leftdom/
```

Important note:

The repository training scripts do not require users to rerun the manuscript's full automated frame filtering pipeline if the split CSV files already point to the filtered frames. For paper-level reproduction, the dominance CSVs should correspond to the same filtered frame lists used in the manuscript experiments.

---

## 12. Summary of Manuscript Study-Level Splits

The following table summarizes the study-level split design used for the three training tasks.

| Task | Unit | Total studies | Train | Validation | Test | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Frame quality assessment | Frame-level supervision, study-level split | 56 | 35 | 9 | 12 | Normal cases only; RCA and LCA |
| RCA occlusion detection | Video/sequence-level supervision | 285 | 180 | 46 | 59 | RCA only |
| Dominance classification | Frame-level prediction, study-level aggregation | 586 | 374 | 94 | 118 | RCA and LCA; filtered informative frames |

The 50-study holdout set and external/challenge subsets used in integrated inference should be represented through separate study-level manifests, not by randomly mixing them into task-level training CSVs.

---

## 13. Integrated Inference Data Structure

Task-level training uses the `<DATA_ROOT>/<task>/DATA_<artery>/<label>/...` structure described above.

Integrated study-level inference uses a different input style. It expects RCA and LCA frame folders grouped by study and sequence/view.

### 13.1 Single RCA/LCA Pair

Use this when a study has one RCA sequence and one LCA sequence.

```text
tmp_data/
├── RCA/
│   └── Study001/
│       └── seq_001/
│           ├── frame_001.png
│           ├── frame_002.png
│           └── ...
└── LCA/
    └── Study001/
        └── seq_001/
            ├── frame_001.png
            ├── frame_002.png
            └── ...
```

Command inputs:

```text
--rca_frame_dir tmp_data/RCA/Study001/seq_001
--lca_frame_dir tmp_data/LCA/Study001/seq_001
```

### 13.2 Multi-Sequence Single-Study Inference

Use this when one study contains multiple RCA and LCA sequences/views.

```text
tmp_data/
├── RCA/
│   └── Study001/
│       ├── seq_001/
│       │   ├── frame_001.png
│       │   └── ...
│       └── seq_002/
│           ├── frame_001.png
│           └── ...
└── LCA/
    └── Study001/
        ├── seq_001/
        │   ├── frame_001.png
        │   └── ...
        └── seq_002/
            ├── frame_001.png
            └── ...
```

Command inputs:

```text
--rca_study_dir tmp_data/RCA/Study001
--lca_study_dir tmp_data/LCA/Study001
```

Supported sequence pairing policies:

```text
trim_to_min
strict_equal
```

`trim_to_min` pairs sorted RCA/LCA sequence folders up to the smaller sequence count and records unpaired sequences in the pairing report.

`strict_equal` raises an error if the number of RCA and LCA sequences differs.

---

## 14. Manifest CSV for Cohort Evaluation

Manifest mode is used for cohort-level integrated inference, including manuscript Table 9-style evaluation.

Required columns:

```csv
study_id,label,rca_study_dir,lca_study_dir
```

Optional column:

```csv
subset
```

Allowed dominance labels:

```text
rightdom
leftdom
```

Example:

```csv
study_id,label,subset,rca_study_dir,lca_study_dir
Study001,rightdom,test,D:\data\RCA\Study001,D:\data\LCA\Study001
Study002,leftdom,test,D:\data\RCA\Study002,D:\data\LCA\Study002
Study003,rightdom,domain_shift,D:\data\RCA\Study003,D:\data\LCA\Study003
```

If a file path contains a comma, wrap that path in quotation marks:

```csv
Study004,rightdom,test,"D:\data,version2\RCA\Study004","D:\data,version2\LCA\Study004"
```

Recommended subset names:

```text
test
holdout
real_distribution
domain_shift
artefacts
bad_quality
uncertainty
occlusions
```

The script does not require these exact subset names, but consistent names make the output easier to compare with the manuscript tables.

If the manifest contains a `subset` column, the script writes subset-level metrics. If the column is absent, only overall metrics are written.

---

## 15. Smoke-Test Data vs Full Paper Data

The repository may include lightweight example split CSVs and/or a small `mini_dataset` for smoke testing.

Smoke-test data is useful for checking:

```text
script execution
CSV parsing
path formatting
model forward passes
evaluation output writing
integrated inference output writing
```

Smoke-test data is not sufficient for:

```text
reproducing paper-level metrics
training final teacher models
training final MTL models
evaluating manuscript tables
comparing backbones or ablations
```

Paper-level reproduction requires:

```text
original CoronaryDominance dataset access
extracted frames organized under <DATA_ROOT>
full study-level split CSVs for all tasks
study-level manifest CSVs for integrated inference cohorts
verified checkpoints, if checkpoint-based reproduction is used
```

---

## 16. Verifying the Dataset

Before training or evaluation, run:

```bash
python scripts/verify_splits.py --data_root <DATA_ROOT> --split_root splits
```

For a local smoke-test dataset:

```bash
python scripts/verify_splits.py --data_root mini_dataset --split_root splits_mini
```

A successful smoke-test verification should report that all checked rows are valid and that no referenced files are missing.

If verification fails, check:

```text
wrong <DATA_ROOT>
wrong <SPLIT_ROOT>
incorrect task folder name
incorrect artery folder name
incorrect label folder name
absolute paths from another machine
CSV filename column includes duplicated folder prefixes
study_id values inconsistent across task CSVs
CSV split/fold values do not match the command arguments
```

---

## 17. Recommended Preparation Workflow

For a new machine, use the following workflow:

1. Obtain the official CoronaryDominance dataset according to its access and license terms.
2. Extract the dataset archives.
3. Read each `.npz` view file and export `pixel_array` frames to image files.
4. Preserve study IDs, artery type, view/series identity, and frame order during frame extraction.
5. Arrange frames under `<DATA_ROOT>` using the task/artery/label structure.
6. Replace any smoke-test split CSVs with the full manuscript split CSVs.
7. Convert machine-specific absolute paths to portable relative paths.
8. Verify all split CSVs:

```bash
python scripts/verify_splits.py --data_root <DATA_ROOT> --split_root splits
```

9. Train or evaluate teacher models.
10. Train or evaluate multi-task models.
11. Use manifest mode for study-level integrated inference and manuscript-style cohort evaluation.

---

## 18. Minimal Checklist Before Running Training

Check the following before running `train_teacher.py` or `train_mtl.py`:

```text
[ ] <DATA_ROOT> exists
[ ] <SPLIT_ROOT> exists
[ ] task folders are named dominance, framequality, occlusion
[ ] artery folders are named DATA_RCA and DATA_LCA
[ ] label folders match the expected label strings
[ ] CSV files contain filename,label,study_id,artery,split,task,fold
[ ] all frames from the same study remain in the same split
[ ] occlusion data exists only for DATA_RCA
[ ] filenames resolve correctly on the current machine
[ ] verify_splits.py passes without missing files
```

---
