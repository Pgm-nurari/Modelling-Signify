# Modelling-Signify

This repository contains the training, evaluation, and testing pipeline for the **HWGAT-E** model on the **INCLUDE** dataset. It is developed as part of the **Major Project** by **Group A8**.

---

## 📁 Project Structure

```
sl-hwgat-main/
├── hwgat/                   # HWGAT model code
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sl-hwgat-main.git
cd sl-hwgat-main
```

### 2. Create a Virtual Environment (Recommended)

Ensure **Python version ≤ 3.11.7** is installed.

```bash
python3 -m venv env
source env/bin/activate    # On Windows: env\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### To Train the Model

```bash
python main.py --mode train --config configs.py
```

### To Evaluate the Model

```bash
python main.py --mode eval --config configs.py
```

### To Test the Model

```bash
python main.py --mode test --config configs.py
```

> Make sure the dataset and required directories are configured correctly inside `configs.py`.

---

## 📦 Dataset Information

- The model is trained on the **INCLUDE** dataset.
- Use the provided script to generate training/testing data splits first:

```bash
python generate_train_csv.py
```

- Ensure dataset paths and CSV structure match the requirements inside the model and config files.

---

## ✅ Notes

- Keep the `hwgat/` folder inside the main project directory: `sl-hwgat-main/`
- Modify hyperparameters and paths using `configs.py`
- All logs, checkpoints, and output directories should be pre-defined or handled in `configs.py`

---

## 👨‍💻 Authors

**Group A8**  
Department of Computer Science  
Amrita Campus
