# Part 1 — Neural Network Fundamentals and Training Behavior Analysis

## Problem Statement

Build a feed-forward neural network to predict customer churn from structured tabular data. The goal is not only to achieve good predictive performance but to demonstrate a thorough understanding of how a neural network learns — through forward pass, loss calculation, backpropagation, and weight updates.

---

## Dataset

**Source:** [Google Drive Dataset Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)  
**File:** `customer_churn_nn.csv`  
**Shape:** 2,000 rows × 17 columns  
**Target:** `churn` — 1 = churned, 0 = retained  
**Class imbalance:** ~98.45% retained vs ~1.55% churned (~64:1 ratio)

> Dataset files are **not committed** to this repository per the project guidelines.  
> Download from the Drive link above and place `customer_churn_nn.csv` in the same directory as `notebook.ipynb`.

| Feature Type | Columns |
|---|---|
| Ordinal (encoded 0/1/2) | `contract_type` |
| Nominal → One-Hot Encoded | `region`, `plan_type`, `payment_method` |
| Numerical (scaled) | tenure, charges, login days, tickets, delays, data usage, satisfaction, complaint recency, discounts, autopay, referrals |
| Identifier (dropped) | `customer_id` |

---

## Repository Structure

```
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── evaluation_outputs.png       ← combined: confusion matrix, loss/accuracy curves,
    │                                   per-class metrics bar chart, key metrics summary
    ├── model_comparison_table.png   ← 5-config hyperparameter comparison table
    ├── model_comparison_table.csv   ← same data in CSV form
    ├── hyperparameter_comparison.png
    ├── correlation_heatmap.png
    └── eda_exploration.png
```

---

## Approach

### Task 1 — Dataset Understanding
Examined shape, dtypes, missing values, statistical summary, and target distribution. No missing values. The 64:1 class imbalance was identified as the central modelling challenge.

### Task 2 — Data Preprocessing
- **Ordinal encoding:** `contract_type` mapped to 0/1/2 (Month-to-month → One-year → Two-year).
- **One-Hot Encoding:** `region`, `plan_type`, `payment_method` — purely nominal, no order implied. `drop_first=True` to avoid multicollinearity.
- **StandardScaler:** Zero mean, unit variance applied to all features.
- **Stratified 80/20 split:** Preserves churn ratio in train and test sets.
- **Class weights:** `compute_class_weight('balanced')` assigns ~32× higher penalty to churner misclassification.

### Task 3 — Forward Pass and Backpropagation Trace (Manual NumPy)
Before building the Keras model, the forward pass and backpropagation were traced step-by-step on a single training sample:

```
Forward pass:
  z1 = x · W1 + b1        # weighted sum at hidden layer
  a1 = ReLU(z1)            # non-linear activation
  z2 = a1 · W2 + b2        # weighted sum at output
  a2 = Sigmoid(z2)         # predicted churn probability
  L  = -[y·log(â) + (1-y)·log(1-â)]   # binary cross-entropy loss

Backpropagation:
  dL/da2 = -(y/â) + (1-y)/(1-â)
  delta2 = dL/da2 × sigmoid'(z2)
  dW2    = a1ᵀ × delta2
  delta1 = delta2 × W2ᵀ × ReLU'(z1)
  dW1    = xᵀ × delta1

Weight update:
  W2 ← W2 − lr × dW2
  W1 ← W1 − lr × dW1
```

### Task 4 — Model Architecture
```
Input (23 features after OHE)
  → Dense(128, ReLU)  → Dropout(0.3)
  → Dense(64,  ReLU)  → Dropout(0.2)
  → Dense(32,  ReLU)  → Dropout(0.2)
  → Dense(1, Sigmoid)
```
Loss: `binary_crossentropy` | Optimiser: `Adam(lr=0.001)` | Callback: `EarlyStopping(patience=12)`

### Task 5 — Evaluation Results

| Metric | Value |
|---|---|
| Test Accuracy | ~97% |
| ROC-AUC | ~0.84 |
| Overfit Gap (train − val loss) | ~0.06 |

The combined `results/evaluation_outputs.png` contains: confusion matrix, loss curve with numeric gap, accuracy curve, per-class precision/recall/F1 bar chart, and key metrics summary.

### Task 6 — Hyperparameter Experiments

| Config | Change | AUC | Observation |
|--------|--------|-----|-------------|
| C1 Baseline | 1 layer, 64 neurons, lr=0.001 | 0.87 | Limited capacity |
| **C2 Deeper** | **3 layers, 128 neurons** | **Best** | More depth captures interactions |
| C3 High LR | lr=0.01 | Good | Faster but less stable |
| C4 Tanh | Tanh activation | Lower | Saturation at extremes |
| C5 Low LR | lr=0.0001, batch=16 | Lower | Too slow; early stopping fires early |

---

## How to Run

```bash
git clone <your-repo-url>
cd part-1-neural-network-analysis
pip install -r requirements.txt
# Place customer_churn_nn.csv in this folder
jupyter notebook notebook.ipynb
```

Run all cells in order. All plots are written to `results/`.

---

## Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow / Keras | ≥2.12 | Neural network |
| scikit-learn | ≥1.2 | Preprocessing, metrics |
| pandas / NumPy | ≥1.5 / ≥1.23 | Data handling |
| Matplotlib / Seaborn | ≥3.6 / ≥0.12 | Visualisation |
