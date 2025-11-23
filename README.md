# GAN Challenge – WGAN-GP Baseline (32×32 Image Generation)

This repository contains a Kaggle notebook solution for the **GAN Challenge**: generating realistic **32×32 images** from a noisy, base64-encoded dataset and submitting **Inception-V3 feature vectors** for FID evaluation.

The pipeline is built around:

- **Robust decoding & cleaning** of the JSONL shards  
- A **WGAN-GP** model with DCGAN-style generator and conv critic  
- **Offline FID** computation using Inception-V3  
- Utility code to generate **1000 samples** and build the required `submission.csv`

---

## 1. Competition Objective

- Train a generative model (GAN, WGAN, Diffusion, etc.) on the provided image shards.
- Reconstruct valid **32×32 RGB** images from base64-encoded JSONL records.
- Generate samples, extract **2048-dim Inception-V3 features** (pool3 layer), and submit:
  ```text
  1000 rows × (1 id + 2048 feature columns)


## 2. Data Pipeline

1. **Decode JSONL shards**
   - Handles `img_b64` / `img64` / `img_64`.
   - Fixes base64 padding, tolerates truncated files.
   - Applies `exif_rot`, `invert`.
   - Converts `"I;16"` → 8-bit grayscale, handles `RGBA` by compositing on white.
   - Ensures final **RGB 32×32** images.

2. **Cleaning**
   - Computes simple grayscale stats (mean, std, near-0/near-1 fraction).
   - Drops “blank-ish” images (almost constant / all black / all white).
   - Keeps the rest as the **real** training set.

3. **NPY dataset**
   - Packs cleaned PNGs into `clean_images_v1_uint8.npy` with shape `(N, 3, 32, 32)`.
   - `Dataset` maps to `float32` in **[-1, 1]** (for `tanh` output).

---

## 3. Model & Training

- **Generator**: DCGAN-style upsampling (ConvTranspose2d + BatchNorm + ReLU → 3×32×32, `Tanh`), `nz=128`.
- **Critic**: Conv-only network (no normalization, no sigmoid).
- **Loss**: WGAN-GP
  - Critic: `-(E[D(real)] - E[D(fake)]) + λ_gp * GP`
  - Generator: `-E[D(fake)]`
- **Hyperparameters**
  - `lambda_gp = 5.0`, `n_critic = 5`
  - Adam (`lr = 1e-4`, `betas = (0.0, 0.9)`)
  - Batch size `128`, ~20–30 epochs
- Saves:
  - Samples: `samples_v1/epoch_XXX.png`
  - Checkpoints: `checkpoints_v1/G_epoch_XXX.pth`

---

## 4. FID & Submission

- Uses **Inception-V3 (ImageNet weights, `fc=Identity`)** to get 2048-dim features.
- Computes offline FID between:
  - features from cleaned real images, and
  - features from generated samples.
- For a chosen epoch (e.g. 20):
  1. Load `G_epoch_020.pth`.
  2. Generate **1000 images**: `dig-000000.png` … `dig-000999.png`.
  3. Extract features and build `submission_epoch_020.csv`:
     - 1000 rows × 2049 columns (`id + f0..f2047`).
