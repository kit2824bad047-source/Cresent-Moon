<div align="center">

# 🌙 Crescent Moon — Binary Classification with an ANN

**A compact neural network that learns to separate two interleaving half-moons — 97.4% test accuracy with just 769 parameters.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Open in Colab](https://img.shields.io/badge/Google%20Colab-ready-F9AB00?logo=googlecolab&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.40%25-brightgreen)

<img src="assets/decision_boundary.png" alt="ANN decision boundary on the make_moons dataset" width="640">

</div>

---

## 📖 Overview

The two-moons problem is a classic benchmark for **non-linearly separable data**: no straight line can split the two classes, so a linear model like logistic regression struggles. This project trains a small **feed-forward Artificial Neural Network (ANN)** with Keras to solve it, and walks through the full machine-learning workflow end to end:

> generate data → visualize → split → scale → build → train → evaluate → visualize the decision boundary → predict

Everything lives in a single, well-commented script (`colab_code.py`) that runs locally or in Google Colab.

## ✨ Highlights

- 🧪 **10,000 synthetic samples** from `sklearn.datasets.make_moons` (noise = 0.20)
- 🧠 **Lightweight ANN** — three hidden layers (32 → 16 → 8) with ReLU, sigmoid output
- ⚖️ **Stratified 80/20 split** and **standardized features** (scaler fit on training data only — no leakage)
- 📊 Full evaluation: accuracy, precision/recall/F1, confusion matrix, learning curves
- 🗺️ **Decision-boundary plot** to see what the network actually learned
- 🔁 **Reproducible** — fixed random seeds for NumPy and TensorFlow
- 💾 Exports the generated dataset to `make_moons_dataset.csv` (auto-downloads in Colab)

## 🏗️ Model Architecture

```
Input (2 features)
      │
Dense(32, ReLU)
      │
Dense(16, ReLU)
      │
Dense(8,  ReLU)
      │
Dense(1, Sigmoid)  →  P(class = 1)
```

| Setting          | Value                                   |
| ---------------- | --------------------------------------- |
| Parameters       | **769** (≈ 3 KB)                        |
| Optimizer        | Adam, learning rate `0.001`             |
| Loss             | Binary cross-entropy                    |
| Epochs           | 150                                     |
| Batch size       | 32                                      |
| Validation split | 20% of the training set                 |
| Decision rule    | predict class 1 if probability ≥ 0.5    |

## 📈 Results

Evaluated on **2,000 held-out test samples** (1,000 per class):

| Metric          | Value      |
| --------------- | ---------- |
| **Accuracy**    | **97.40%** |
| Test loss       | 0.0694     |
| Precision (avg) | 0.97       |
| Recall (avg)    | 0.97       |
| F1-score (avg)  | 0.97       |

| Class   | Precision | Recall | F1-score | Support |
| ------- | :-------: | :----: | :------: | :-----: |
| Class 0 |   0.98    |  0.97  |   0.97   |  1000   |
| Class 1 |   0.97    |  0.98  |   0.97   |  1000   |

> Numbers come from a run with TensorFlow 2.21 on CPU. Seeds are fixed, but results can vary slightly across TensorFlow versions and hardware.

### Visual results

<table>
  <tr>
    <td align="center"><b>Dataset</b><br><img src="assets/dataset.png" width="380" alt="make_moons dataset"></td>
    <td align="center"><b>Confusion matrix</b><br><img src="assets/confusion_matrix.png" width="380" alt="Confusion matrix"></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><b>Training &amp; validation curves</b><br><img src="assets/training_curves.png" width="780" alt="Training and validation accuracy and loss"></td>
  </tr>
</table>

**Confusion matrix:** 969 + 979 samples classified correctly, 31 + 21 misclassified — errors are balanced across both classes, which is expected since they overlap only along the noisy boundary where the moons meet.

**Learning curves:** accuracy and loss converge within roughly 10 epochs and stay flat afterwards, with training and validation curves close together — no sign of overfitting.

## 🚀 Getting Started

### Option 1 — Google Colab (no setup)

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Paste the contents of `colab_code.py` into a cell (or upload the file and run `%run colab_code.py`).
3. Run it. The dataset CSV downloads automatically at the end.

### Option 2 — Run locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Crescent_moon_BICLASS_ANN.git
cd Crescent_moon_BICLASS_ANN

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python colab_code.py
```

Plot windows open one after another — close each to continue. Training takes a few minutes on a CPU.

## 🔬 What the Script Does

| Step | Description |
| :--: | ----------- |
| 1–2  | Import libraries, fix random seeds, generate 10,000 moon-shaped samples |
| 3    | Scatter-plot the two classes |
| 4    | Stratified 80/20 train/test split |
| 5    | Standardize features with `StandardScaler` (fit on train only) |
| 6    | Build and compile the Keras `Sequential` model |
| 7    | Train for 150 epochs with a 20% validation split |
| 8–10 | Evaluate: loss/accuracy, classification report, confusion matrix |
| 11   | Plot training/validation accuracy and loss |
| 12   | Render the decision boundary over a 300×300 grid |
| 13   | Predict the class of a new sample, `[0.5, 0.2]` |
| 14–15| Save the dataset as CSV (and download it in Colab) |

### Predicting a new point

```python
sample = np.array([[0.5, 0.2]])
probability = model.predict(scaler.transform(sample), verbose=0)[0][0]
print(int(probability >= 0.5), probability)   # → 1, ≈ 0.59
```

> Remember to pass new data through the **same fitted `scaler`** used during training.

## 📁 Project Structure

```
Crescent_moon_BICLASS_ANN/
├── colab_code.py        # Complete pipeline: data → model → evaluation → plots
├── requirements.txt     # Python dependencies
├── assets/              # Result figures used in this README
└── README.md
```

## 🛠️ Tech Stack

- **[TensorFlow / Keras](https://www.tensorflow.org/)** — neural network
- **[scikit-learn](https://scikit-learn.org/)** — dataset generation, splitting, scaling, metrics
- **[NumPy](https://numpy.org/)** & **[pandas](https://pandas.pydata.org/)** — data handling
- **[Matplotlib](https://matplotlib.org/)** — visualization

## 🎛️ Experiment Ideas

- Change `noise` in `make_moons` to make the problem easier or harder
- Try different depths/widths, or swap ReLU for `tanh` / `LeakyReLU`
- Add `Dropout` or `EarlyStopping` to cut training time
- Compare against logistic regression, SVM (RBF), or Random Forest baselines
- Tune the classification threshold (`0.5`) using a precision–recall curve

## 🤝 Contributing

Suggestions and pull requests are welcome. Fork the repo, create a feature branch, and open a PR describing your change.

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">

If you found this useful, consider giving the repo a ⭐

</div>
