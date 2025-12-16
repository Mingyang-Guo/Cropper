# Cropper

# Cropper: Attention-Guided Ensemble Model for Crop Disease Detection 🌿

Cropper is a lightweight and interpretable deep ensemble framework for detecting crop diseases using leaf images. It integrates multiple convolutional backbones with attention mechanisms to deliver high accuracy, visual interpretability, and edge-device deployability, particularly in data-scarce agricultural environments.

## 🌟 Key Features

- 🔁 Multi-backbone ensemble (InceptionV3, ResNet34, EfficientNetV2-S)
- 🧠 Attention-guided feature refinement (Channel & Spatial attention)
- 🖼️ Visualization with class-discriminative attention maps
- 🌾 Supports multiple crops: rice, betel, cabbage, Chinese cabbage
- ⚙️ Includes benchmark models: PlantDet, Guava, LeafNext for comparison
- 📉 Ablation-friendly architecture and reproducible evaluation protocol


## 📁 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `Cropper_Rice.py` | Main Cropper training + evaluation script for **rice dataset** |
| `Cropper_Betel.py` | Main Cropper training + evaluation script for **betel dataset** |
| `Cropper_Cabbage.py` | Main Cropper training + evaluation script for **cabbage dataset** |
| `Cropper_ChineseCabbage.py` | Main Cropperscript for **Chinese cabbage dataset** |
| `Guava.py` | Guava baseline implementation |
| `LeafNext_Betel.py` | LeafNext baseline implementation for **betel** |
| `LeafNext_Rice.py` | LeafNext baseline implementation for **rice** |
| `PlantDet_Betel.py` | PlantDet benchmark model for **betel** |
| `PlantDet_Rice.py` | PlantDet benchmark model for **rice** |
| `Visual.py` | Script for generating and saving attention visualizations (heatmaps) |
| `dataset_Info.xlsx` | Summary of all datasets used (image count, labels, splits, etc.) |
| `Results.xlsx` | Evaluation metrics and performance comparison tables |
| `README.md` | You are here. Project overview and usage instructions. |

---

## 🧪 Datasets

The model supports **4 plant leaf datasets** (publicly available):

- **Rice**: Multi-class (6 disease types) 
- **Betel**: Binary classification (healthy vs. unhealthy)
- **Cabbage**: Binary classification (nutrient deficiency vs. healthy)
- **Chinese Cabbage**: Binary classification (Botanical Leaf Spot)

The linkings of these datasets can be found in the paper.

Dataset info and splits stats are documented in `dataset_Info.xlsx`.

> 📦 Note: Please organize your image folders following the structure:
> ```
> dataset/
> ├── rice/
> │   ├── Bacterial_leaf_blight/
> │   ├── Brown_spot/
> │   └── ...
> ├── betel/
> │   ├── Healthy/
> │   └── Unhealthy/
> ...
> ```

# Environment 
Shown in the .yaml file 
