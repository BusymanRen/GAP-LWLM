```markdown
# GAP-LWLM

Official implementation of the three-stage training pipeline for GAP-LWLM (Geometry-Aware Prompt Tuning for Cross-Scene Localization)[cite: 6].

---

## 1. Environment Setup

Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install pytorch-lightning timm h5py numpy scipy

```

---

## 2. Dataset Structure

Sample data has already been prepared in the repository with the following layout:

```text
├── scene_1/
│   ├── scene_1.npy
│   ├── pretrain/
│   │   ├── train.npy
│   │   └── val_indices.npy
│   ├── tune/
│   │   ├── train.npy
│   │   ├── val_indices.npy
│   │   └── test_indices.npy
│   └── BS1_...sc_...at_...M/
│   │       ├── rss.npy
│   │       ├── los_probability_heatmap.npy
│   │       └── *.h5py
│   └── ...

```

---

## 3. Training & Running

Execute the three stages sequentially.

**Stage 1: Pretraining**

```bash
python ./train_scene.py --task pretrain_mix --model_path ./pretrained_model/testmix --is_frozen 0 --is_load 0 --depth 4

```

**Stage 2: Fine-Tuning**

```bash
python ./train_scene.py --task SingleBSLoc --model_path ./tune_model/ --is_frozen 0 --is_load 0 --depth 4 --load_pretrained_mdl_path ./pretrained_model/testmix/testepoch=00.ckpt

```

**Stage 3: Prompt Tuning**

```bash
python ./train_scene.py --task prompt_loc --is_frozen 0 --is_load 0 --depth 4 --load_pretrained_mdl_path ./pretrained_model/testmix/testepoch=01-01.ckpt --model_path ./prompt_model/

```

```

```