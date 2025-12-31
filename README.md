# AutoJudge: Programming Problem Difficulty Predictor

## Project Overview
AutoJudge is a machine learning system that automatically predicts the difficulty of programming problems based on their textual descriptions. The system classifies problems as Easy, Medium, or Hard and predicts a continuous difficulty score between 0 and 10.

## 🚀 Features

### 🤖 Smart Difficulty Prediction
Class Prediction: Classifies problems as Easy, Medium, or Hard

Score Prediction: Provides numerical score (0-10 scale)

Confidence Scoring: Shows model confidence for each prediction

### 🔍 Advanced Feature Analysis
Algorithm Detection: Identifies 100+ algorithms (DP, Graphs, Trees, etc.)

Mathematical Analysis: Detects math symbols, formulas, and complexity

Structural Analysis: Analyzes problem structure, constraints, examples

Text Complexity: Word count, sentence structure, readability metrics

### 🎯 Multi-Model Architecture
Random Forest Classifier: For difficulty classification

Random Forest Regressor: For numerical scoring

TF-IDF Vectorizer: Text feature extraction

Ensemble Approach: Combines multiple models for accuracy

### 🌐 User-Friendly Interface
Clean, modern web interface

Real-time predictions

Detailed results breakdown

Sample problems for testing

Mobile-responsive design

## 🏗️ System Architecture

User Input → Text Processing → Feature Extraction → ML Prediction → Results Display
    │              │                 │                 │              │
    │         Clean text        Extract 24+     Classification  Visualize
    │       Remove noise        features       & Regression    with insights


## 🔧 Technical Stack
Backend: Flask (Python)

Frontend: HTML5, CSS3, JavaScript, Bootstrap

ML Framework: Scikit-learn, Pandas, NumPy

Text Processing: NLTK, Regex

Visualization: Matplotlib, Seaborn

## Step-by-Step Setup
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Install ML model dependencies
cd ml_model
pip install -r requirements.txt

# Install web app dependencies
cd ../web_app
pip install -r requirements.txt

cd ml_model
python train.py

This will:

✅ Load/Create training data

✅ Extract 24 advanced features

✅ Train classification & regression models

✅ Save models to saved_models/

✅ Generate performance reports

cd web_app
python app.py

http://localhost:5000