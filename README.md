# 🚀 AutoJudge — Programming Problem Difficulty Predictor

AutoJudge is a machine learning–powered system that **automatically predicts the difficulty of programming problems** using only their textual descriptions.  
It classifies problems as **Easy / Medium / Hard** and assigns a **continuous difficulty score between 0 and 10**, along with confidence estimates.

---

## 📌 Project Overview

Online coding platforms typically assign difficulty levels using manual judgment and user feedback.  
AutoJudge aims to **automate this process** by analyzing the **structure, language, and mathematical content** of programming problem statements.

### 🎯 What AutoJudge Predicts
- **Difficulty Class** → Easy / Medium / Hard  
- **Difficulty Score** → Continuous value in **[0, 10]**  
- **Prediction Confidence** → Model certainty for each output

---

## ✨ Key Features

### 🤖 Smart Difficulty Prediction
- Multi-class classification (Easy / Medium / Hard)
- Numerical difficulty scoring (0–10 scale)
- Confidence estimation for predictions

---

### 🔍 Advanced Feature Analysis
- **Algorithm Detection**  
  Detects 100+ algorithmic patterns (DP, Graphs, Trees, Greedy, etc.)

- **Mathematical Analysis**  
  Counts symbols, formulas, and mathematical expressions

- **Structural Analysis**  
  Examines constraints, examples, and problem layout

- **Text Complexity Metrics**  
  Word count, sentence structure, and readability indicators

---

### 🎯 Multi-Model Architecture
- **Random Forest Classifier** → Difficulty class
- **Random Forest Regressor** → Difficulty score
- **TF-IDF Vectorizer** → Text representation
- **Ensemble-style feature design** for higher robustness

---

### 🌐 User-Friendly Web Interface
- Clean and modern UI
- Real-time predictions
- Detailed result breakdown
- Sample problems for testing
- Mobile-responsive design

---

## 🏗️ System Architecture

User Input
↓
Text Cleaning & Normalization
↓
Feature Extraction (TF-IDF + Numeric Features)
↓
ML Models (Classification & Regression)
↓
Prediction + Confidence
↓
Web Visualization


---

## ⚙️ Step-by-Step Setup

### 🔹 1. Create Virtual Environment

#### Windows
python -m venv venv
venv\Scripts\activate

#### macOS / Linux
python3 -m venv venv
source venv/bin/activate

### 2. Install ML Dependencies
cd ml_model
pip install -r requirements.txt

### 3. Install Web App Dependencies
cd ../web_app
pip install -r requirements.txt

### 4. Train the Models
cd ../ml_model
python train.py

### 5. Run the Web Application
cd ../web_app
python app.py

## Open your browser at:

👉 http://localhost:5000