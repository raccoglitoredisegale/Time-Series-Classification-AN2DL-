# 🏴‍☠️ Pirate Pain Prediction | AN2DL Challenge 1

**Course:** Artificial Neural Networks and Deep Learning (Politecnico di Milano) 

---

## 🧭 Project Overview

This project addresses the first challenge of the AN2DL course: classifying the pain status of subjects (pirates and ordinary folk) based on multivariate time-series data. The goal is to predict the subject’s true pain status (`no_pain`, `low_pain`, `high_pain`) using motion data captured from body joints and sensor aggregations.

Our solution implements a **GRU-based architecture with Attention** and a dedicated **Static Feature Embedding path**, achieving a mean weighted F1-score of **0.9381**.

---

## 👥 The Team

**Alberto Occhipinti**  
 
**Giulia Putelli**  
 
**Matteo Morini**  
 
**Tommaso Rossetti** 

---

## 📂 Dataset & Task

### The Data

The dataset contains recordings of subjects identified by `sample_index`. Each sample includes 180 time steps with the following features:

* **Temporal Features:** 31 continuous measurements of body joint angles (`joint_00`–`joint_30`) and 4 pain survey aggregations (`pain_survey_1`–`4`).
* **Static Features:** 3 categorical subject characteristics (`n_legs`, `n_hands`, `n_eyes`).

### The Goal

Perform **Time Series Classification** to assign one of three labels to each sequence:

1. `no_pain`
2. `low_pain`
3. `high_pain`

> **Constraint:** Usage of pretrained models was strictly **forbidden**. All architectures were trained from scratch.
> 
> 

---

## 🧠 Our Solution

We developed a custom **RecurrentClassifierWithAttention** model designed to process temporal and static data in parallel.

### 1. Data Preprocessing
 
**Windowing:** We utilized a sliding window approach with a **Window Size of 44** and a **Stride of 22**, creating overlapping sequences to maximize training data.
 
**Normalization:** Min-Max normalization was applied to all 35 time-series features, fitted only on the training folds to prevent leakage.
 
**Categorical Mapping:** Static features (e.g., "one+peg_leg") were mapped to integers for embedding.

### 2. Model Architecture

The architecture consists of three main components:
 
**Temporal Path:** A 2-layer **GRU** (128 hidden units, 0.5 dropout) processes the time-series data.
 
**Static Path:** Parallel embedding layers process the categorical features, which are then concatenated and projected via an MLP.
 
**Attention Mechanism:** An Attention Encoder layer uses the GRU's final hidden state as a query to create a weighted context vector from all time steps, focusing on the most relevant parts of the sequence.



### 3. Training Strategy 
**Optimizer:** `AdamW` with weight decay ().

**Loss Function:** Weighted `CrossEntropyLoss` (to handle class imbalance) + **Label Smoothing** (0.1).

**Validation:** 5-Fold **Stratified Group K-Fold**. This was crucial to ensure that all data from a single subject remained in one split, preventing data leakage.

---

## 📊 Results

The model was evaluated using the weighted **F1-Score**.

We employed an ensemble strategy for the final submission, averaging the softmax probabilities of the 5 best models (one from each fold) and using a majority vote aggregation per subject.

| Metric | Score |
| --- | --- |
| **Mean Validation F1** | <br>**0.9381 ± 0.014** 

 |
| **Best Fold F1** | <br>**0.9631** 

 |

---

## 🛠️ Repository Structure

```text
├── data/                  # Dataset files (train/test csv)
├── notebooks/             # Jupyter notebooks for training and analysis
├── models/                # Saved model checkpoints
├── src/                   # Source code for preprocessing and model definitions
├── FinalReport.pdf        # Detailed project report
└── README.md              # Project documentation

```

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/thegradientdescenters/Time-Series-Classification-AN2DL.git

```


2. Install dependencies:
```bash
pip install -r requirements.txt

```


3. Run the training notebook in the `notebooks/` directory to reproduce the results.
