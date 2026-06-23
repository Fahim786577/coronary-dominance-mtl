# Coronary Dominance Multi-Task Learning

This repository provides the implementation and reproducibility package for the coronary dominance classification framework described in:

**Multi-Task Learning with Knowledge Distillation for Robust Coronary Dominance Classification in Invasive Angiography**

The project implements a unified multi-task learning pipeline for invasive coronary angiography (ICA) frames/videos. The model jointly learns:

1. Coronary dominance classification
2. Right coronary artery (RCA) occlusion detection
3. Frame quality assessment

The repository supports baseline multi-task learning, multi-teacher distillation, practical TwoPhase-inspired optimization, RCA-to-LCA transfer learning, task-level evaluation, and integrated study-level inference.

---

## Repository Contents

```text
coronary-dominance-mtl/
├── scripts/
│   ├── verify_splits.py
│   ├── smoke_test_model.py
│   ├── smoke_test_mtd.py
│   ├── train_teacher.py
│   ├── train_mtl.py
│   ├── evaluate_mtl.py
│   ├── predict_mtl.py
│   └── run_integrated_inference.py
├── splits/
│   ├── dominance/
│   ├── framequality/
│   └── occlusion/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
├── DATASET.md
├── REPRODUCIBILITY.md
├── requirements.txt
└── README.md
```

---

## Tested Environment

The experiments were developed and tested with:

```text
Python 3.9.8
PyTorch 2.6.0
CUDA 12.4
NVIDIA GeForce RTX 3090 GPU
```

The code intentionally uses a lightweight dependency set. Most CSV, JSON, and metric export operations are implemented using the Python standard library.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Fahim786577/coronary-dominance-mtl.git
cd coronary-dominance-mtl
```

Create and activate a Python environment:

```bash
python -m venv <ENV_NAME>
<ENV_NAME>\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, install the PyTorch build that matches your CUDA version. The experiments in this work used PyTorch 2.6.0 with CUDA 12.4.

---

## Main Tasks

The repository supports three binary classification tasks:

| Task                     | Internal task name               | Classes                         |
| ------------------------ | -------------------------------- | ------------------------------- |
| Dominance classification | `dominance`                      | `rightdom`, `leftdom`           |
| Frame quality assessment | `framequality` / `frame_quality` | `noninformative`, `informative` |
| RCA occlusion detection  | `occlusion`                      | `nonoccluded`, `occluded`       |

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

Note: in the model output dictionary, frame quality is represented as `frame_quality`.

---

## Supported Backbones

The multi-task model supports:

```text
resnet18
mobilenet_v2
densenet121
```
The current implementation supports only the backbones listed above. Additional CNN or transformer backbones can be added by extending `src/models/backbones.py`, defining the appropriate grayscale input adaptation, and updating the corresponding feature dimension used by the task heads.

The main model is:

```text
CoronaryTemporalMTL
```

It contains:

```text
shared CNN backbone
+ occlusion LSTM video head
+ frame quality classification head
+ dominance classification head
```

RCA training uses all three tasks:

```text
occlusion
frame_quality
dominance
```

LCA training uses:

```text
frame_quality
dominance
```

Occlusion detection is restricted to RCA views.

---

## Dataset and Splits

The repository includes example split CSVs under `splits/` to demonstrate the expected directory structure and CSV format used by the training and evaluation scripts.

**Important:** the split CSVs currently included in this public repository are smoke-test/example splits, not the full original paper splits. They are intended to verify code execution, path formatting, and pipeline behavior. Reproducing the paper-level results requires the full dataset-specific split CSVs and study-level manifests used in the manuscript experiments.

Expected split structure:

```text
splits/
├── dominance/
│   ├── DATA_RCA/labels/
│   └── DATA_LCA/labels/
├── framequality/
│   ├── DATA_RCA/labels/
│   └── DATA_LCA/labels/
└── occlusion/
    └── DATA_RCA/labels/
```

The split CSVs use the enriched format:

```csv
filename,label,study_id,artery,split,task,fold
```

The `filename` column should point to the corresponding frame path. Depending on the user’s local machine, this may need to be updated to match the local dataset location.

More dataset and preprocessing details are provided in [`DATASET.md`](DATASET.md).

---

## Verify Split Files

Before training or evaluation, verify that the split CSVs are readable and that the referenced frame files exist:

```bash
python scripts/verify_splits.py --data_root <DATA_ROOT> --split_root splits
```

For the small local smoke-test dataset used during development:

```bash
python scripts/verify_splits.py --data_root mini_dataset --split_root splits_mini
```

---

## Smoke Tests

Run the model smoke test:

```bash
python scripts/smoke_test_model.py
```

Run the multi-teacher distillation smoke test:

```bash
python scripts/smoke_test_mtd.py
```

These tests check model construction, basic forward passes, and distillation loss logic.

---

## Teacher Training

Single-task teacher models are trained independently for each task before multi-teacher distillation. See `DATASET.md` for the expected `<DATA_ROOT>` layout.

Teacher checkpoints are saved under:

```text
outputs/teachers/<task>/DATA_<artery>/fold_<fold>/<backbone>/best.pt
```
The supported teacher tasks are:

```text
dominance
framequality
occlusion
```
Occlusion detection is only defined for RCA views.

### Dominance Teacher

```bash
python scripts/train_teacher.py ^
  --task dominance ^
  --artery <artery> ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_root outputs/teachers ^
  --epochs 100 ^
  --batch_size 8 ^
  --pretrained true ^
  --device cuda
```

Expected teacher checkpoint structure:

```text
outputs/teachers/<task>/DATA_<artery>/fold_<fold>/<backbone>/best.pt
```

For example, if `--artery RCA` is used:

```text
outputs/teachers/dominance/DATA_RCA/fold_1/resnet18/best.pt
```

Teacher outputs are raw logits.

---

### Frame-Quality Teacher

```bash
python scripts/train_teacher.py ^
  --task framequality ^
  --artery <artery> ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_root outputs/teachers ^
  --epochs 100 ^
  --batch_size 8 ^
  --pretrained true ^
  --device cuda
```

Expected checkpoint:

For example, if `--artery RCA` is used:

```text
outputs/teachers/framequality/DATA_RCA/fold_1/resnet18/best.pt
```

### RCA Occlusion Teacher

```bash
python scripts/train_teacher.py ^
  --task occlusion ^
  --artery RCA ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_root outputs/teachers ^
  --epochs 100 ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained true ^
  --device cuda
```

Expected checkpoint:

```text
outputs/teachers/occlusion/DATA_RCA/fold_1/resnet18/best.pt
```

Teacher outputs are raw logits. During multi-teacher distillation, these logits are converted to softened probability distributions using temperature-scaled KL divergence.

---

## Multi-Task Training

The output folder is selected automatically based on these options.
 
The script does not use a `--mode` argument; the training configuration is selected using `--use_mtd` and `--use_twophase`.

| Configuration        | `--use_mtd` | `--use_twophase` | Output folder               |
| -------------------- | ----------- | ---------------- | --------------------------- |
| Baseline MTL         | `false`     | `false`          | `outputs/mtl/baseline/`     |
| MTL + MTD            | `true`      | `false`          | `outputs/mtl/mtd/`          |
| MTL + TwoPhase       | `false`     | `true`           | `outputs/mtl/twophase/`     |
| MTL + MTD + TwoPhase | `true`      | `true`           | `outputs/mtl/mtd_twophase/` |


### Baseline Multi-Task Learning:

```bash
python scripts/train_mtl.py ^
  --artery RCA ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_dir outputs/mtl ^
  --epochs 100 ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained true ^
  --device cuda ^
  --use_mtd false ^
  --use_twophase false
```
Expected output folder:

```text
outputs/mtl/baseline/DATA_RCA/fold_1/resnet18/
```

---

## Multi-Teacher Distillation

MTL with multi-teacher distillation trains the multi-task student using both ground-truth supervision and teacher-guided distillation.

Teacher checkpoints should exist under:

```text
outputs/teachers/<task>/DATA_<artery>/fold_<fold>/<backbone>/best.pt
```

For RCA MTD training, the active teachers are:

```text
outputs/teachers/occlusion/DATA_RCA/fold_1/resnet18/best.pt
outputs/teachers/framequality/DATA_RCA/fold_1/resnet18/best.pt
outputs/teachers/dominance/DATA_RCA/fold_1/resnet18/best.pt
```

Command:

```bash
python scripts/train_mtl.py ^
  --artery RCA ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_dir outputs/mtl ^
  --teacher_root outputs/teachers ^
  --teacher_checkpoint_name best.pt ^
  --epochs 100 ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained true ^
  --device cuda ^
  --use_mtd true ^
  --use_twophase false ^
  --mtd_temperature 4.0 ^
  --mtd_alpha_occlusion 0.1 ^
  --mtd_alpha_frame_quality 0.1 ^
  --mtd_alpha_dominance 0.1
```

Expected output folder:

```text
outputs/mtl/mtd/DATA_RCA/fold_1/resnet18/
```

Default MTD settings:

```text
temperature = 4.0
alpha_occlusion = 0.1
alpha_frame_quality = 0.1
alpha_dominance = 0.1
```

MTD is a training-time strategy. At evaluation time, the saved checkpoint is loaded as a standard multi-task model.

---

## Practical TwoPhase-Inspired Optimization

The repository includes a practical priority-aware TwoPhase-inspired optimization option.

**Important note:**

This implementation is a practical approximation inspired by Two-Phase task-priority optimization. It is not a full reproduction of the original channel-wise connection-strength implementation.

### MTL + TwoPhase

```bash
python scripts/train_mtl.py ^
  --artery RCA ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_dir outputs/mtl ^
  --epochs 100 ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained true ^
  --device cuda ^
  --use_mtd false ^
  --use_twophase true
```

Expected output folder:

```text
outputs/mtl/twophase/DATA_RCA/fold_1/resnet18/
```

### MTL + MTD + TwoPhase

```bash
python scripts/train_mtl.py ^
  --artery RCA ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_dir outputs/mtl ^
  --teacher_root outputs/teachers ^
  --teacher_checkpoint_name best.pt ^
  --epochs 100 ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained true ^
  --device cuda ^
  --use_mtd true ^
  --use_twophase true ^
  --mtd_temperature 4.0 ^
  --mtd_alpha_occlusion 0.1 ^
  --mtd_alpha_frame_quality 0.1 ^
  --mtd_alpha_dominance 0.1
```

Expected output folder:

```text
outputs/mtl/mtd_twophase/DATA_RCA/fold_1/resnet18/
```

---

## RCA to LCA Transfer Learning

The LCA model can be initialized from a trained RCA checkpoint using:

```text
--transfer_from_checkpoint
```

The recommended transfer scope is:

```text
--transfer_load_scope shared_and_common_heads
```

This loads the shared backbone plus the common frame-quality and dominance heads. Occlusion-specific parameters are skipped for LCA.

For LCA training, the active tasks are:

```text
frame_quality
dominance
```

Occlusion detection is not trained for LCA.

### Baseline RCA-to-LCA Transfer

```bash
python scripts/train_mtl.py ^
  --artery LCA ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_dir outputs/mtl ^
  --epochs 50 ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained true ^
  --device cuda ^
  --use_mtd false ^
  --use_twophase false ^
  --transfer_from_checkpoint outputs/mtl/baseline/DATA_RCA/fold_1/resnet18/best.pt ^
  --transfer_from_artery RCA ^
  --transfer_load_scope shared_and_common_heads
```

Expected output folder:

```text
outputs/mtl/baseline_transfer/DATA_LCA/fold_1/resnet18/
```

### MTD RCA-to-LCA Transfer

For LCA MTD transfer training, the required teachers are:

```text
outputs/teachers/framequality/DATA_LCA/fold_1/resnet18/best.pt
outputs/teachers/dominance/DATA_LCA/fold_1/resnet18/best.pt
```

Command:

```bash
python scripts/train_mtl.py ^
  --artery LCA ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_dir outputs/mtl ^
  --teacher_root outputs/teachers ^
  --teacher_checkpoint_name best.pt ^
  --epochs 50 ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained true ^
  --device cuda ^
  --use_mtd true ^
  --use_twophase false ^
  --mtd_temperature 4.0 ^
  --mtd_alpha_frame_quality 0.1 ^
  --mtd_alpha_dominance 0.1 ^
  --transfer_from_checkpoint outputs/mtl/mtd/DATA_RCA/fold_1/resnet18/best.pt ^
  --transfer_from_artery RCA ^
  --transfer_load_scope shared_and_common_heads
```

Expected output folder:

```text
outputs/mtl/mtd_transfer/DATA_LCA/fold_1/resnet18/
```

### TwoPhase RCA-to-LCA Transfer

```bash
python scripts/train_mtl.py ^
  --artery LCA ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_dir outputs/mtl ^
  --epochs 50 ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained true ^
  --device cuda ^
  --use_mtd false ^
  --use_twophase true ^
  --transfer_from_checkpoint outputs/mtl/twophase/DATA_RCA/fold_1/resnet18/best.pt ^
  --transfer_from_artery RCA ^
  --transfer_load_scope shared_and_common_heads
```

Expected output folder:

```text
outputs/mtl/twophase_transfer/DATA_LCA/fold_1/resnet18/
```

### MTD + TwoPhase RCA-to-LCA Transfer

This is the full proposed LCA transfer configuration.

```bash
python scripts/train_mtl.py ^
  --artery LCA ^
  --fold 1 ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --output_dir outputs/mtl ^
  --teacher_root outputs/teachers ^
  --teacher_checkpoint_name best.pt ^
  --epochs 50 ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained true ^
  --device cuda ^
  --use_mtd true ^
  --use_twophase true ^
  --mtd_temperature 4.0 ^
  --mtd_alpha_frame_quality 0.1 ^
  --mtd_alpha_dominance 0.1 ^
  --transfer_from_checkpoint outputs/mtl/mtd_twophase/DATA_RCA/fold_1/resnet18/best.pt ^
  --transfer_from_artery RCA ^
  --transfer_load_scope shared_and_common_heads
```

Expected output folder:

```text
outputs/mtl/mtd_twophase_transfer/DATA_LCA/fold_1/resnet18/
```

The following transfer override flags are available for debugging only and should generally not be used in reproducibility commands:

```text
--allow_transfer_backbone_mismatch true
--allow_transfer_fold_mismatch true
```

---

## Task-Level Evaluation

Use `evaluate_mtl.py` for task-level evaluation. This evaluates each active task separately.

For RCA, the active tasks are:

```text
occlusion
frame_quality
dominance
```

For LCA, the active tasks are:

```text
frame_quality
dominance
```

### RCA Task-Level Evaluation

```bash
python scripts/evaluate_mtl.py ^
  --artery RCA ^
  --fold 1 ^
  --split test ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --mode baseline ^
  --checkpoint_name best.pt ^
  --output_dir outputs/evaluation ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained false ^
  --device cuda
```

Expected output folder:

```text
outputs/evaluation/baseline/DATA_RCA/fold_1/resnet18/test/
```

### LCA Task-Level Evaluation

```bash
python scripts/evaluate_mtl.py ^
  --artery LCA ^
  --fold 1 ^
  --split test ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --mode mtd_twophase_transfer ^
  --checkpoint_name best.pt ^
  --output_dir outputs/evaluation ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained false ^
  --device cuda
```

Expected output folder:

```text
outputs/evaluation/mtd_twophase_transfer/DATA_LCA/fold_1/resnet18/test/
```

Outputs:

```text
metrics.json
metrics.csv
predictions.csv
```

Metrics include accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1-score, MCC, confusion matrix values, and sample count.

---

## Prediction Export

Use `predict_mtl.py` to export task-level predictions.

```bash
python scripts/predict_mtl.py ^
  --artery RCA ^
  --fold 1 ^
  --split test ^
  --backbone resnet18 ^
  --data_root <DATA_ROOT> ^
  --split_root splits ^
  --mode baseline ^
  --checkpoint_name best.pt ^
  --output_dir outputs/predictions ^
  --batch_size 8 ^
  --clip_length 15 ^
  --pretrained false ^
  --device cuda
```

Expected output folder:

```text
outputs/predictions/baseline/DATA_RCA/fold_1/resnet18/test/
```

Outputs:

```text
predictions.csv
```

---

## Integrated Study-Level Inference

The integrated inference system follows the clinical decision pathway:

```text
Start with RCA
↓
Run RCA occlusion detection
↓
If RCA is non-occluded:
    use RCA frames
Else:
    switch to LCA frames
↓
Run frame quality filtering
↓
Keep informative frames
↓
Run dominance classification
↓
Majority vote
↓
Final study-level dominance prediction
```

This is implemented in:

```text
scripts/run_integrated_inference.py
```

---

## Single RCA/LCA Pair Inference

Use this mode when you want to perform integrated coronary dominance inference on a single RCA/LCA angiography pair. The system accepts one RCA sequence and one LCA sequence, performs RCA occlusion detection, artery selection, frame-quality filtering, dominance classification, and majority-vote aggregation, and then outputs a final study-level dominance prediction.

Recommended full-system checkpoints:

```text
RCA:
outputs/mtl/mtd_twophase/DATA_RCA/fold_1/resnet18/best.pt

LCA:
outputs/mtl/mtd_twophase_transfer/DATA_LCA/fold_1/resnet18/best.pt
```
For `--input_mode single_pair`, each input directory should directly contain frames from one RCA sequence and one LCA sequence.

For example, if a study has one RCA sequence and one LCA sequence, then the single-pair command should use the sequence folders directly.

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


Command:

```bash
python scripts/run_integrated_inference.py ^
  --input_mode single_pair ^
  --rca_frame_dir tmp_data/RCA/Study001/seq_001 ^
  --lca_frame_dir tmp_data/LCA/Study001/seq_001 ^
  --rca_checkpoint outputs/mtl/mtd_twophase/DATA_RCA/fold_1/resnet18/best.pt ^
  --lca_checkpoint outputs/mtl/mtd_twophase_transfer/DATA_LCA/fold_1/resnet18/best.pt ^
  --backbone resnet18 ^
  --output_dir outputs/integrated_single_pair ^
  --device cuda ^
  --clip_length 15 ^
  --image_size 512
```

Outputs:

```text
final_prediction.json
frame_predictions.csv
```

The final JSON contains the selected artery route, RCA occlusion probability, number of informative frames, dominance votes, final predicted class, predicted label, and final confidence.

---

## Multi-Sequence Single-Study Inference

Use this mode when you want to perform integrated study-level inference on a study that contains multiple RCA and LCA angiographic sequences/views. The system automatically pairs the available RCA and LCA sequences according to the selected pairing policy, performs integrated inference on each sequence pair, and aggregates the resulting pair-level predictions into a final study-level coronary dominance prediction.

Expected input structure:

```text
tmp_data/
├── RCA/
│   ├── Study001/
│   │   ├── seq_001/
│   │   │   ├── frame_001.png
│   │   │   └── ...
│   │   └── seq_002/
│   │       ├── frame_001.png
│   │       └── ...
│   └── Study002/
│       ├── seq_001/
│       │   ├── frame_001.png
│       │   └── ...
│       └── seq_002/
│           ├── frame_001.png
│           └── ...
└── LCA/
    ├── Study001/
    │   ├── seq_001/
    │   │   ├── frame_001.png
    │   │   └── ...
    │   └── seq_002/
    │       ├── frame_001.png
    │       └── ...
    └── Study002/
        ├── seq_001/
        │   ├── frame_001.png
        │   └── ...
        └── seq_002/
            ├── frame_001.png
            └── ...
```

Command:

```bash
python scripts/run_integrated_inference.py ^
  --input_mode multi_sequence_study ^
  --rca_study_dir tmp_data/RCA/Study001 ^
  --lca_study_dir tmp_data/LCA/Study001 ^
  --rca_checkpoint outputs/mtl/mtd_twophase/DATA_RCA/fold_1/resnet18/best.pt ^
  --lca_checkpoint outputs/mtl/mtd_twophase_transfer/DATA_LCA/fold_1/resnet18/best.pt ^
  --backbone resnet18 ^
  --output_dir outputs/integrated_study ^
  --device cuda ^
  --clip_length 15 ^
  --image_size 512 ^
  --sequence_pair_policy trim_to_min
```

Sequence pairing policies:

```text
trim_to_min
strict_equal
```

`trim_to_min` pairs sorted RCA/LCA sequences up to the smaller sequence count and records extra unpaired sequences in the pairing report.

`strict_equal` raises an error when RCA and LCA sequence counts differ.

Outputs:

```text
study_final_prediction.json
pair_predictions.csv
pair_frame_predictions.csv
sequence_pairing_report.csv
```


---

## Manifest Cohort Evaluation

Manifest mode evaluates multiple studies and supports the paper's Table 9-style cohort evaluation. This is the recommended mode for reproducing study-level integrated inference results across cohorts such as test, holdout, real distribution, domain shift, artefacts, bad quality, and uncertainty.

The script does not hard-code cohort names. If a `subset` column is present, metrics are computed separately for each subset. A manifest may contain one subset or multiple subsets.

Required manifest columns:

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

Example manifest CSV:

```csv
study_id,label,subset,rca_study_dir,lca_study_dir
Study001,rightdom,test,D:\data\RCA\Study001,D:\data\LCA\Study001
Study002,leftdom,test,D:\data\RCA\Study002,D:\data\LCA\Study002
Study003,rightdom,domain_shift,D:\data\RCA\Study003,D:\data\LCA\Study003
```

Command:

```bash
python scripts/run_integrated_inference.py ^
  --input_mode manifest ^
  --manifest_csv <MANIFEST_CSV> ^
  --rca_checkpoint outputs/mtl/mtd_twophase/DATA_RCA/fold_1/resnet18/best.pt ^
  --lca_checkpoint outputs/mtl/mtd_twophase_transfer/DATA_LCA/fold_1/resnet18/best.pt ^
  --backbone resnet18 ^
  --output_dir outputs/integrated_inference/manifest_eval ^
  --device cuda ^
  --clip_length 15 ^
  --image_size 512 ^
  --sequence_pair_policy trim_to_min ^
  --invalid_study_policy skip
```

Invalid study handling:

```text
skip
error
```

`skip` records invalid studies in the output CSV and excludes them from metric computation.

`error` stops execution when an invalid study is encountered.

Outputs:

```text
integrated_predictions.csv
integrated_metrics.csv
integrated_metrics.json
subset_metrics.csv
all_pair_predictions.csv
all_pair_frame_predictions.csv
all_sequence_pairing_report.csv
```

`subset_metrics.csv` is written only when the manifest contains a `subset` column.

Table 9-style reproduction requires a study-level manifest containing the same evaluation cohorts used in the manuscript experiments.

---

## Checkpoints

Checkpoints are not committed directly to this repository.

Expected checkpoint locations:

```text
outputs/teachers/<task>/DATA_<artery>/fold_<fold>/<backbone>/best.pt
outputs/mtl/<mode>/DATA_<artery>/fold_<fold>/<backbone>/best.pt
```

Checkpoint release is currently under verification to ensure compatibility with the reorganized codebase and will be documented separately if provided.
---

## License and Citation

Please cite the original CoronaryDominance dataset paper if you use the dataset:

Kruzhilov, I. et al. CoronaryDominance: Angiogram dataset for coronary dominance classification. Scientific Data 12, 341 (2025). https://doi.org/10.1038/s41597-025-04676-8

If you use or refer to the TwoPhase-inspired optimization component, please also cite:

Jeong et al. Quantifying Task Priority for Multi-Task Optimization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.
