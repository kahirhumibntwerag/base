# Benchmarking Deep Learning Architectures for Super-Resolution of Solar Images

## 🚀 Overview
Super-resolution of solar images is crucial for studying fine-scale structures in the solar corona. This project explores deep learning models to enhance low-resolution (LR) solar images into high-resolution (HR) outputs. It implements and evaluates CNNs (RRDB), GANs (ESRGAN), and Diffusion Models (LDM, ELDM) for enhancing extreme ultraviolet (EUV) solar images at the 171 Å wavelength. Models are evaluated using standard metrics (SSIM, PSNR) and a novel Statistical Fidelity (SF) metric. The repository also provides interactive visualization tools to compare LR and super-resolved images.

## 📥 Installation
To set up the environment and dependencies, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/super_resolution_solar.git
cd super_resolution_solar

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use "venv\Scripts\activate"

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model checkpoints (for inference & visualization)
mkdir -p checkpoints
curl -L -o checkpoints/RRDB.ckpt "https://huggingface.co/yourusername/super_resolution_solar/resolve/main/RRDB.ckpt"
curl -L -o checkpoints/ESRGAN.ckpt "https://huggingface.co/yourusername/super_resolution_solar/resolve/main/ESRGAN.ckpt"
curl -L -o checkpoints/LDM.ckpt "https://huggingface.co/yourusername/super_resolution_solar/resolve/main/LDM.ckpt"
curl -L -o checkpoints/ELDM.ckpt "https://huggingface.co/yourusername/super_resolution_solar/resolve/main/ELDM.ckpt"
```

These commands will install all required packages and automatically download the pre-trained model weights into the `checkpoints/` folder. If you prefer to train models from scratch, you may skip downloading these checkpoints.

## 😍 Visual Results
### Demo on Real-world SR

[<img src="figs/half.png" height="213px"/>](https://imgsli.com/MzUzNjIw) 

For more comparisons, please refer to our paper for details.

## 📂 Repository Structure
```
📂 super_resolution_solar/        # Project root directory
├── 📂 data/                      # Dataset files (e.g., train.pt, val.pt, test.pt)
├── 📂 src/                       # Source code for models and utilities (RRDB, ESRGAN, LDM, ELDM implementations)
├── 📂 checkpoints/               # Model checkpoints (pre-trained or saved during training)
├── 📂 results/                   # Outputs, figures, and visualizations
├── 📄 config.yml                 # Configuration file for training (hyperparameters, paths)
├── 📄 requirements.txt           # Python dependencies
├── 📄 train.py                   # Training script for models
├── 📄 inference.py               # Script for model inference on test data
├── 📄 visualize_results.py       # Script to generate qualitative LR vs SR image comparisons
├── 📄 evaluate.py                # Evaluation script to compute metrics on test set
└── 📄 README.md                  # Project documentation
```

## 📊 Model Comparison
### 🔍 SF vs SSIM Trade-off
We evaluate each model's performance by Structural Similarity Index (SSIM) (higher is better for structure) and Statistical Fidelity (SF) (lower is better for fine-detail accuracy). Generally, higher SSIM indicates better structural preservation, while lower SF indicates better consistency in fine details.

## Model Performance

| Model | SSIM ↑ | PSNR (dB) ↑ | SF ↓ |
|-------|---------|-------------|-------|
| RRDB (CNN) | 0.9771 | 39.91 | 0.01920 |
| ESRGAN (GAN) | 0.9735 | 41.03 | 0.00383 |
| LDM (Diffusion) | 0.9641 | 37.75 | 0.00749 |
| ELDM (Ours) | 0.9687 | 38.45 | 0.00512 |

*Table: Quantitative performance of each model on the test set. RRDB achieves the highest SSIM (best structural similarity) but also the highest SF (more fine-detail differences), whereas ESRGAN achieves the lowest SF (best detail fidelity) with a slightly lower SSIM.*

### SF vs. SSIM Scatter Plot
The following graph illustrates the trade-off between **Statistical Fidelity (SF)** and **Structural Similarity Index (SSIM)** for different models:

![SF vs SSIM](figs/comparison.png)

*Each point represents a model's performance, showing the inherent trade-off between structural similarity and statistical fidelity.*

## 💪 Training the Models
```bash
# Example: Train the ESRGAN model for 150 epochs with batch size 8
python train.py --model ESRGAN --epochs 150 --batch_size 8
```

Available `--model` options: `RRDB`, `ESRGAN`, `LDM`, `ELDM`. Model and training hyperparameters can be adjusted via `config.yml` or overridden with command-line arguments.

## 🔬 Dataset
We train and evaluate on a dataset of 5,000 solar images from NASA's Solar Dynamics Observatory (SDO) taken at the 171 Å EUV wavelength. Data preprocessing includes:
- **Downsampling:** Reduce resolution from 4K to 512×512.
- **Instrument corrections:** Apply exposure and sensor degradation corrections.
- **Normalization:** Scale pixel values to [-1, 1] for stable model training.

## 💟 Future Work
- **Multi-wavelength Training:** Extend dataset to ~50,000 images including other EUV wavelengths.
- **Efficiency Improvements:** Explore multi-GPU training and model pruning.
- **Hybrid Models:** Investigate combining GAN and diffusion models for improved performance.

## 📝 Citation
If you use this code or models in your research, please cite the thesis:
```bibtex
@article{elsheikh2025superresolution,
  title={Benchmarking Deep Learning Architectures for Super-Resolution of Solar Images},
  author={Elsheikh, Mohamed Hisham Mahmoud},
  journal={Bachelor Thesis},
  year={2025}
}
```

## 🗃️ License
This project is released under the MIT License. See the LICENSE file for details.

