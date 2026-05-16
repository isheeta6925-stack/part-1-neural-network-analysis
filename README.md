# Part 1 — Neural Network Fundamentals and Training Behavior Analysis

## Problem Statement

This project builds and analyses a feed-forward neural network to predict customer churn from structured, tabular data. The focus is not only on achieving a good model but on understanding *how* the neural network learns — through forward pass, loss calculation, backpropagation, and parameter updates.

---

## Dataset

**Source:** [Google Drive Dataset Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)  
**File used:** `customer_churn_nn.csv`  
**Rows / Columns:** 2,000 rows × 17 columns  
**Target variable:** `churn` — 1 = churned, 0 = retained  
**Class distribution:** ~98.45% retained, ~1.55% churned (severe imbalance)

**Feature categories:**

| Type | Columns |
|------|---------|
| Categorical | `region`, `plan_type`, `contract_type`, `payment_method` |
| Numerical | `tenure_months`, `monthly_charges_inr`, `avg_login_days_per_month`, `support_tickets_last_90_days`, `payment_delay_days`, `data_usage_gb`, `satisfaction_score`, `last_complaint_days_ago`, `discount_percent`, `autopay_enabled`, `referral_count` |
| Identifier (dropped) | `customer_id` |

---

## Approach

### Task 1 — Dataset Understanding
Loaded the dataset and examined shape, data types, missing values, statistical summaries, and target distribution. No missing values were found. The target class is highly imbalanced at approximately 64:1 (retained vs churned), which directly shapes every modelling decision downstream.

### Task 2 — Data Preprocessing
- **Categorical encoding:** Label Encoding applied to `region`, `plan_type`, `contract_type`, and `payment_method`.
- **Scaling:** `StandardScaler` applied to all features — zero mean, unit variance — to prevent high-magnitude features from dominating gradient updates.
- **Train/test split:** 80/20 stratified split to preserve churn ratio in both sets.
- **Class weights:** Computed via `sklearn.utils.class_weight.compute_class_weight('balanced')`. The churned class received a weight of ~32, penalising missed churners more heavily during training.

### Task 3 — Neural Network Architecture
Built a feed-forward neural network in TensorFlow/Keras:

```
Input (15 features)
  → Dense(128, ReLU) → Dropout(0.3)
  → Dense(64,  ReLU) → Dropout(0.3)
  → Dense(1, Sigmoid)
```

- **Loss:** Binary Cross-Entropy  
- **Optimiser:** Adam (lr=0.001)  
- **Activation (hidden):** ReLU — avoids vanishing gradients, computationally efficient  
- **Activation (output):** Sigmoid — produces calibrated probability in [0, 1]  
- **Regularisation:** Dropout (0.3) to reduce overfitting

### Task 4 — Training and Evaluation
Trained with `EarlyStopping` (patience=12) on 15% validation split. Key results for the best model:

| Metric | Value |
|--------|-------|
| Test Accuracy | ~94% |
| ROC-AUC | ~0.83 |
| Macro F1 | ~0.53 |

The confusion matrix and training curves are saved under `results/`.

**Interpretation:** High accuracy on this dataset is misleading due to class imbalance. The ROC-AUC of 0.83 is a more honest indicator — it measures ranking quality independent of the decision threshold. The model correctly distinguishes churners from non-churners in 83% of pair-wise comparisons.

### Task 5 — Hyperparameter Experimentation

Five configurations were tested. Key findings:

| Config | Change | ROC-AUC | Observation |
|--------|--------|---------|-------------|
| C1 Baseline | 1 layer, 64 neurons, lr=0.001 | 0.67 | Underpowered for interaction learning |
| **C2 Deeper** | **3 layers, 128 neurons** | **0.90** | **Best overall — more capacity helps** |
| C3 Higher LR | lr=0.01 | 0.90 | Fast convergence but less stable test accuracy |
| C4 Tanh | Tanh activation | 0.74 | Slightly weaker than ReLU on this data |
| C5 Small LR | lr=0.0001, batch=16 | 0.82 | Very slow; early stopping triggered prematurely |

Config 2 achieved the best ROC-AUC. Higher depth with ReLU and moderate learning rate is the most effective combination.

### Task 6 — Final Reflection

#### Weights and Biases
Weights are the learnable scalars on each neuron connection; biases are additive offsets. Together they parameterise the linear part of each neuron's computation: `z = Wx + b`. Through backpropagation, the optimiser adjusts these values to minimise the loss, progressively mapping input features to accurate churn probabilities.

#### Why Activation Functions Are Required
Without activation functions, stacked dense layers reduce to a single linear transformation regardless of depth. Activation functions (ReLU here) introduce non-linearity, allowing the network to learn curved decision boundaries and capture feature interactions — for instance, the combined effect of low satisfaction score *and* a month-to-month contract on churn risk.

#### Effect of Learning Rate
- **Too high (lr=0.01):** Overshoots the loss minimum; training becomes unstable; validation accuracy lower and less consistent.
- **Too low (lr=0.0001):** Extremely slow convergence; early stopping fires before the model reaches a good solution.
- **Optimal (lr=0.001):** Smooth, steady convergence to a good minimum.

#### Underfitting vs Overfitting
The best model shows no significant overfitting — training and validation losses track closely throughout training. The low F1 on the minority class is not overfitting but rather *underfitting on the rare class*, a consequence of having very few churner examples (only 31 out of 2,000). Class-weight balancing partially addresses this, but more representative data or synthetic oversampling (SMOTE) would be needed for further improvement.

---

## Repository Structure

```
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── eda_exploration.png
    ├── correlation_heatmap.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── hyperparameter_comparison.png
    ├── model_comparison_table.png
    └── model_comparison_table.csv
```

---

## How to Run

```bash
# Clone the repository
git clone <your-repo-url>
cd part-1-neural-network-analysis

# Install dependencies
pip install -r requirements.txt

# Place dataset file in the project root (do not upload to GitHub)
# Download from: https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

# Launch notebook
jupyter notebook notebook.ipynb
```

---

## Key Libraries

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | Neural network construction and training |
| scikit-learn | Preprocessing, splitting, evaluation metrics |
| pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualisation |
