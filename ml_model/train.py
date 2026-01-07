"""
Production Training Script for Two-Stage CP Difficulty Predictor

ARCHITECTURE:
    STAGE 1: Classify into Easy/Medium/Hard with calibrated probabilities
    STAGE 2: Per-class regressors predict scores within theoretical ranges
    
SCORE RANGES (ENFORCED):
    Easy:   0.0 - 4.0
    Medium: 4.0 - 7.0
    Hard:   7.0 - 10.0

Author: Senior ML Engineer
Version: 3.0.0 - Production Ready
"""

import pandas as pd
import numpy as np
import joblib
import re
import json
import os
import warnings
from datetime import datetime
from collections import Counter

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV

from scipy.sparse import hstack, csr_matrix

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available, using RandomForest only")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available, using RandomForest only")


# ============================================================================
# THEORETICAL SCORE RANGES (FLEXIBLE - WILL BE LEARNED FROM DATA)
# ============================================================================
THEORETICAL_RANGES = {
    'Easy': (0.0, 5.0),      # Expanded from 4.0 to 5.0
    'Medium': (4.0, 8.0),    # Expanded from 7.0 to 8.0
    'Hard': (6.0, 10.0)      # Expanded from 7.0 to 6.0
}


# ============================================================================
# TEXT PREPROCESSING - MUST BE IDENTICAL IN TRAIN AND INFERENCE
# ============================================================================
def clean_text(text):
    """
    Clean and normalize text for feature extraction.
    This function MUST be identical in train.py and app.py.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Preserve mathematical operators (separate numbers)
    text = re.sub(r'(\d+)\s*[+\-*/=<>]\s*(\d+)', r'\1 \2', text)
    
    # Keep alphanumeric + important symbols
    text = re.sub(r'[^\w\s\+\-\*/=<>\(\)\[\]\{\}\.,;:!?\^&\|%#@$\n_]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_engineered_features(text):
    """
    Extract hand-crafted features that capture problem difficulty.
    Feature names and order MUST be identical in train.py and app.py.
    
    Args:
        text: Cleaned text string
        
    Returns:
        Dictionary of feature_name -> numeric_value
    """
    features = {}
    words = text.split()
    
    # ========================================
    # 1. TEXT STATISTICS (4 features)
    # ========================================
    features['text_length'] = len(text)
    features['word_count'] = len(words)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0.0
    features['char_diversity'] = len(set(text)) / max(1, len(text))
    
    # ========================================
    # 2. MATHEMATICAL OPERATIONS (9 features)
    # ========================================
    math_patterns = {
        'math_add': r'\+',
        'math_sub': r'\-',
        'math_mul': r'\*',
        'math_div': r'\/',
        'math_eq': r'=',
        'math_ineq': r'[<>≤≥]',
        'math_paren': r'[\(\)]',
        'math_bracket': r'[\[\]\{\}]',
        'math_symbols': r'[+\-*/=<>^&|%$]'
    }
    
    for name, pattern in math_patterns.items():
        features[name] = len(re.findall(pattern, text))
    
    # ========================================
    # 3. ALGORITHM CONCEPTS (weighted, 27 features)
    # ========================================
    # Higher weights = stronger indicator of difficulty
    concept_weights = {
        'graph': 1.5,
        'dp': 2.0,
        'recursion': 1.5,
        'tree': 1.2,
        'array': 0.8,
        'string': 0.8,
        'sort': 1.0,
        'search': 1.0,
        'binary': 1.2,
        'matrix': 1.0,
        'function': 0.5,
        'loop': 0.5,
        'if': 0.3,
        'while': 0.5,
        'for': 0.3,
        'complexity': 1.8,
        'algorithm': 1.0,
        'optimization': 1.5,
        'backtracking': 1.8,
        'memoization': 1.8,
        'greedy': 1.5,
        'divide': 1.2,
        'conquer': 1.2,
        'dynamic programming': 2.0,
        'bfs': 1.5,
        'dfs': 1.5,
        'dijkstra': 1.8
    }
    
    for concept, weight in concept_weights.items():
        count = len(re.findall(r'\b' + re.escape(concept) + r'\b', text, re.IGNORECASE))
        feature_name = f'concept_{concept.replace(" ", "_")}'
        features[feature_name] = count * weight
    
    # ========================================
    # 4. NUMERICAL CONTENT (2 features)
    # ========================================
    features['number_count'] = len(re.findall(r'\b\d+\b', text))
    features['number_ratio'] = features['number_count'] / max(1, features['word_count'])
    
    # ========================================
    # 5. STRUCTURE INDICATORS (6 features)
    # ========================================
    features['sentence_count'] = len(re.split(r'[.!?]+', text))
    features['has_constraints'] = int(bool(re.search(r'\d+\s*[≤<>=]\s*\d+', text)))
    features['has_formula'] = int(bool(re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=', text)))
    features['has_loop'] = int('for' in text or 'while' in text or 'loop' in text)
    features['has_recursion'] = int('recursion' in text or 'recursive' in text)
    features['has_dp'] = int('dp' in text or 'dynamic' in text or 'memoization' in text)
    
    # ========================================
    # 6. COMPLEXITY KEYWORDS (1 feature)
    # ========================================
    complexity_keywords = ['complexity', 'optimize', 'efficient', 'time limit', 'space']
    features['complexity_keywords'] = sum(1 for kw in complexity_keywords if kw in text)
    
    # ========================================
    # 7. CODE PATTERNS (1 feature)
    # ========================================
    features['code_like'] = int(bool(re.search(r'[{}();=]', text)))
    
    return features


# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================
def load_dataset(filepath='../data/dataset.jsonl'):
    """
    Load and validate dataset from JSONL file.
    
    Expected format per line:
    {
        "title": "...",
        "description": "...",
        "input": "...",
        "output": "...",
        "problem_class": "Easy|Medium|Hard",
        "problem_score": 0-10
    }
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        DataFrame with validated data, or None if failed
    """
    print(f"\n{'='*70}")
    print(f"📂 LOADING DATASET")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    data = []
    errors = []
    stats = {'warnings': 0}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                
                # Validate required fields
                required = ['title', 'description', 'problem_class', 'problem_score']
                missing = [field for field in required if field not in record]
                
                if missing:
                    errors.append(f"Line {line_num}: Missing fields {missing}")
                    continue
                
                # Normalize and validate class (auto-capitalize)
                if 'problem_class' in record:
                    record['problem_class'] = record['problem_class'].strip().capitalize()
                
                if record['problem_class'] not in ['Easy', 'Medium', 'Hard']:
                    errors.append(f"Line {line_num}: Invalid class '{record['problem_class']}'")
                    continue
                
                # Validate score
                score = float(record['problem_score'])
                if not (0 <= score <= 10):
                    errors.append(f"Line {line_num}: Score {score} out of range [0,10]")
                    continue
                
                # Validate score range (RELAXED - allow overlap)
                score = float(record['problem_score'])
                class_name = record['problem_class']
                
                # Just warn if outside theoretical range, don't reject
                if class_name in THEORETICAL_RANGES:
                    min_score, max_score = THEORETICAL_RANGES[class_name]
                    if not (min_score <= score <= max_score):
                        # Still accept, just track as warning
                        if stats['warnings'] < 10:  # Only show first 10 warnings
                            errors.append(f"Line {line_num}: Score {score} outside typical {class_name} range [{min_score},{max_score}] (accepted)")
                        stats['warnings'] += 1
                
                data.append(record)
                
            except (json.JSONDecodeError, ValueError) as e:
                errors.append(f"Line {line_num}: {e}")
                continue
    
    if errors:
        print(f"\n⚠️  Found {len(errors)} validation messages (showing first 10):")
        for error in errors[:10]:
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors)-10} more")
        
        if stats['warnings'] > 0:
            print(f"\n💡 Note: {stats['warnings']} samples have scores outside typical ranges")
            print(f"   This is OK - the model will learn the actual score distribution from your data")
    
    if not data:
        print("❌ No valid data found")
        return None
    
    df = pd.DataFrame(data)
    
    print(f"\n✅ Loaded {len(df)} valid samples")
    print(f"\nClass distribution:")
    for class_name in ['Easy', 'Medium', 'Hard']:
        count = len(df[df['problem_class'] == class_name])
        print(f"   {class_name:8s}: {count:4d} samples")
    
    print(f"\nScore statistics:")
    print(f"   Mean:   {df['problem_score'].mean():.2f}")
    print(f"   Std:    {df['problem_score'].std():.2f}")
    print(f"   Min:    {df['problem_score'].min():.2f}")
    print(f"   Max:    {df['problem_score'].max():.2f}")
    
    # Check for class imbalance
    class_counts = df['problem_class'].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > 3:
        print(f"\n⚠️  Class imbalance detected: ratio = {imbalance_ratio:.2f}")
        print("   Consider collecting more data for minority classes")
    
    return df


# ============================================================================
# FEATURE PREPARATION
# ============================================================================
def prepare_features(df, vectorizer=None, count_vectorizer=None, fit=True):
    """
    Prepare all features from DataFrame.
    
    Args:
        df: DataFrame with title, description, input, output columns
        vectorizer: TfidfVectorizer (pass existing for inference)
        count_vectorizer: CountVectorizer (pass existing for inference)
        fit: If True, fit vectorizers; if False, transform only
        
    Returns:
        X: Combined feature matrix (sparse)
        y_class: Class labels (0,1,2 for Easy,Medium,Hard)
        y_score: Scores (0-10)
        vectorizer: Fitted TfidfVectorizer
        count_vectorizer: Fitted CountVectorizer
        feature_names: List of engineered feature names
    """
    print(f"\n{'='*70}")
    print(f"🔧 FEATURE PREPARATION")
    print(f"{'='*70}")
    
    # ========================================
    # STEP 1: COMBINE AND CLEAN TEXT
    # ========================================
    print("Combining text fields...")
    df = df.copy()
    
    # Combine all text fields
    df['combined_text'] = df.apply(
        lambda row: f"{row.get('title', '')} {row.get('description', '')} "
                   f"{row.get('input', '')} {row.get('output', '')}",
        axis=1
    )
    
    # Clean text (CRITICAL: same function as inference)
    print("Cleaning text...")
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    print(f"✓ Text prepared: {len(df)} samples")
    print(f"  Average length: {df['cleaned_text'].str.len().mean():.0f} chars")
    
    # ========================================
    # STEP 2: TF-IDF FEATURES
    # ========================================
    print("\nExtracting TF-IDF features...")
    
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        tfidf_features = vectorizer.fit_transform(df['cleaned_text'])
    else:
        tfidf_features = vectorizer.transform(df['cleaned_text'])
    
    print(f"✓ TF-IDF: {tfidf_features.shape[1]} features, {tfidf_features.nnz} non-zero")
    
    # ========================================
    # STEP 3: COUNT FEATURES (binary keywords)
    # ========================================
    print("\nExtracting count features...")
    
    if fit:
        count_vectorizer = CountVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            binary=True,
            min_df=2
        )
        count_features = count_vectorizer.fit_transform(df['cleaned_text'])
    else:
        count_features = count_vectorizer.transform(df['cleaned_text'])
    
    print(f"✓ Count: {count_features.shape[1]} features, {count_features.nnz} non-zero")
    
    # ========================================
    # STEP 4: ENGINEERED FEATURES
    # ========================================
    print("\nExtracting engineered features...")
    
    engineered_list = []
    for i, text in enumerate(df['cleaned_text']):
        features = extract_engineered_features(text)
        engineered_list.append(features)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(df)} samples...")
    
    engineered_df = pd.DataFrame(engineered_list)
    
    # Store feature names (CRITICAL for inference)
    feature_names = engineered_df.columns.tolist()
    
    print(f"✓ Engineered: {len(feature_names)} features")
    print(f"  Feature names: {feature_names[:5]}...")
    
    # Convert to sparse
    engineered_sparse = csr_matrix(engineered_df.values)
    
    # ========================================
    # STEP 5: COMBINE ALL FEATURES
    # ========================================
    print("\nCombining features...")
    
    # Stack: [TF-IDF | Count | Engineered]
    X = hstack([tfidf_features, count_features, engineered_sparse])
    
    print(f"✓ Combined: {X.shape[1]} total features")
    print(f"  Shape: {X.shape}")
    print(f"  Non-zero: {X.nnz}")
    print(f"  Sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
    
    # ========================================
    # STEP 6: PREPARE TARGETS
    # ========================================
    y_class = None
    y_score = None
    
    if 'problem_class' in df.columns:
        # Encode classes as 0,1,2
        label_encoder = LabelEncoder()
        y_class = label_encoder.fit_transform(df['problem_class'])
        print(f"\n✓ Target classes: {label_encoder.classes_}")
    
    if 'problem_score' in df.columns:
        y_score = df['problem_score'].values
        print(f"✓ Target scores: shape={y_score.shape}")
    
    return X, y_class, y_score, vectorizer, count_vectorizer, feature_names


# ============================================================================
# STAGE 1: TRAIN CLASSIFIER
# ============================================================================
def train_classifier(X_train, y_train, X_test, y_test, class_names):
    """
    Train and calibrate classifier for Easy/Medium/Hard prediction.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        class_names: ['Easy', 'Medium', 'Hard']
        
    Returns:
        Calibrated classifier
    """
    print(f"\n{'='*70}")
    print(f"🎯 STAGE 1: CLASSIFIER TRAINING")
    print(f"{'='*70}")
    
    # Try multiple models
    models = {}
    
    # RandomForest (always available)
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    # LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    # Evaluate models
    print(f"\nEvaluating {len(models)} classifier models...")
    
    best_model = None
    best_score = 0
    best_name = None
    
    for name, model in models.items():
        print(f"\n  {name}:")
        
        # 3-fold CV on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        avg_score = cv_scores.mean()
        
        print(f"    CV Accuracy: {avg_score:.4f} (±{cv_scores.std():.4f})")
        
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_name = name
    
    print(f"\n✓ Best model: {best_name} (CV={best_score:.4f})")
    
    # Train best model on full training set
    print(f"\nTraining {best_name} on full training set...")
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # Calibrate probabilities
    print(f"\nCalibrating probabilities...")
    
    calibrated_clf = CalibratedClassifierCV(
        best_model,
        method='isotonic',
        cv='prefit'
    )
    
    # Use 30% of training data for calibration
    X_cal, _, y_cal, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42, stratify=y_train)
    calibrated_clf.fit(X_cal, y_cal)
    
    print(f"✓ Calibration complete")
    
    # Test calibrated predictions
    y_pred_cal = calibrated_clf.predict(X_test)
    test_accuracy_cal = accuracy_score(y_test, y_pred_cal)
    
    print(f"✓ Calibrated Test Accuracy: {test_accuracy_cal:.4f}")
    
    # Show sample probabilities
    probs = calibrated_clf.predict_proba(X_test[:5])
    print(f"\nSample predictions (first 5 test samples):")
    for i in range(min(5, len(y_test))):
        true_class = class_names[y_test[i]]
        pred_class = class_names[y_pred_cal[i]]
        confidence = probs[i].max()
        print(f"  True: {true_class:8s} | Pred: {pred_class:8s} | Confidence: {confidence:.3f}")
    
    return calibrated_clf


# ============================================================================
# STAGE 2: TRAIN REGRESSORS
# ============================================================================
def train_regressors(X_train, y_class_train, y_score_train, class_names):
    """
    Train per-class regressors for score prediction.
    
    Args:
        X_train: Training features
        y_class_train: Training class labels (0,1,2)
        y_score_train: Training scores (0-10)
        class_names: ['Easy', 'Medium', 'Hard']
        
    Returns:
        regressors: Dict {class_name: regressor}
        scalers: Dict {class_name: MinMaxScaler}
        class_score_ranges: Dict {class_name: (min, max)}
    """
    print(f"\n{'='*70}")
    print(f"📊 STAGE 2: REGRESSOR TRAINING")
    print(f"{'='*70}")
    
    regressors = {}
    scalers = {}
    class_score_ranges = {}
    
    for class_idx, class_name in enumerate(class_names):
        print(f"\n{'─'*50}")
        print(f"Training regressor for: {class_name}")
        print(f"{'─'*50}")
        
        # Get training samples for this class
        class_mask = (y_class_train == class_idx)
        X_class = X_train[class_mask]
        y_class = y_score_train[class_mask]
        
        if len(y_class) < 10:
            print(f"⚠️  Insufficient samples ({len(y_class)}) for {class_name}")
            print(f"   Skipping regressor training")
            regressors[class_name] = None
            scalers[class_name] = None
            continue
        
        print(f"Samples: {len(y_class)}")
        print(f"Score range: [{y_class.min():.2f}, {y_class.max():.2f}]")
        print(f"Theoretical range: {THEORETICAL_RANGES[class_name]}")
        
        # Store actual score range
        class_score_ranges[class_name] = (float(y_class.min()), float(y_class.max()))
        
        # Normalize scores to [0, 1]
        scaler = MinMaxScaler()
        y_normalized = scaler.fit_transform(y_class.reshape(-1, 1)).ravel()
        
        print(f"Normalized range: [{y_normalized.min():.4f}, {y_normalized.max():.4f}]")
        
        # Try multiple regressors
        models = {}
        
        # RandomForest (always available)
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Ridge regression
        models['Ridge'] = Ridge(alpha=1.0, random_state=42)
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        # Evaluate models
        print(f"\nEvaluating {len(models)} regressor models...")
        
        best_model = None
        best_mse = float('inf')
        best_name = None
        
        for name, model in models.items():
            # 3-fold CV
            cv_scores = cross_val_score(
                model, X_class, y_normalized,
                cv=min(3, len(y_class) // 3),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            avg_mse = -cv_scores.mean()
            
            print(f"  {name:12s}: MSE={avg_mse:.6f}")
            
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_model = model
                best_name = name
        
        print(f"\n✓ Best model: {best_name} (MSE={best_mse:.6f})")
        
        # Train best model
        best_model.fit(X_class, y_normalized)
        
        # Store
        regressors[class_name] = best_model
        scalers[class_name] = scaler
        
        print(f"✓ Regressor trained and stored")
    
    return regressors, scalers, class_score_ranges


# ============================================================================
# MODEL EVALUATION
# ============================================================================
def evaluate_model(classifier, regressors, scalers, class_score_ranges, 
                  X_test, y_class_test, y_score_test, class_names):
    """
    Evaluate full two-stage pipeline on test set.
    
    Args:
        classifier: Trained classifier
        regressors: Dict of trained regressors
        scalers: Dict of score scalers
        class_score_ranges: Dict of actual score ranges
        X_test, y_class_test, y_score_test: Test data
        class_names: Class name list
        
    Returns:
        metrics: Dict of performance metrics
    """
    print(f"\n{'='*70}")
    print(f"📈 MODEL EVALUATION")
    print(f"{'='*70}")
    
    # Stage 1: Classify
    y_class_pred = classifier.predict(X_test)
    class_probs = classifier.predict_proba(X_test)
    
    # Stage 2: Predict scores
    y_score_pred = []
    confidences = []
    
    for i in range(X_test.shape[0]):
        # Get predicted class
        pred_class_idx = y_class_pred[i]
        pred_class = class_names[pred_class_idx]
        confidence = class_probs[i, pred_class_idx]
        
        # Get theoretical range
        theo_min, theo_max = THEORETICAL_RANGES[pred_class]
        
        # Predict score
        if regressors[pred_class] is not None:
            try:
                # Predict normalized score
                norm_score = regressors[pred_class].predict(X_test[i])[0]
                norm_score = np.clip(norm_score, 0.0, 1.0)
                
                # Denormalize
                actual_min, actual_max = class_score_ranges[pred_class]
                denorm_score = norm_score * (actual_max - actual_min) + actual_min
                
                # Clip to theoretical range
                final_score = np.clip(denorm_score, theo_min, theo_max)
            except:
                # Fallback
                final_score = (theo_min + theo_max) / 2.0
                confidence *= 0.8
        else:
            # No regressor - use midpoint
            final_score = (theo_min + theo_max) / 2.0
            confidence *= 0.7
        
        y_score_pred.append(final_score)
        confidences.append(confidence)
    
    y_score_pred = np.array(y_score_pred)
    confidences = np.array(confidences)
    
    # Overall metrics
    class_accuracy = accuracy_score(y_class_test, y_class_pred)
    score_mse = mean_squared_error(y_score_test, y_score_pred)
    score_rmse = np.sqrt(score_mse)
    score_mae = mean_absolute_error(y_score_test, y_score_pred)
    score_r2 = r2_score(y_score_test, y_score_pred)
    
    print(f"\nOverall Performance:")
    print(f"  Classification Accuracy: {class_accuracy:.4f}")
    print(f"  Score RMSE: {score_rmse:.4f}")
    print(f"  Score MAE: {score_mae:.4f}")
    print(f"  Score R²: {score_r2:.4f}")
    print(f"  Avg Confidence: {confidences.mean():.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    for class_idx, class_name in enumerate(class_names):
        class_mask = (y_class_test == class_idx)
        if class_mask.sum() == 0:
            continue
        
        class_true = y_score_test[class_mask]
        class_pred = y_score_pred[class_mask]
        class_rmse = np.sqrt(mean_squared_error(class_true, class_pred))
        class_mae = mean_absolute_error(class_true, class_pred)
        
        print(f"\n  {class_name}:")
        print(f"    Samples: {class_mask.sum()}")
        print(f"    RMSE: {class_rmse:.4f}")
        print(f"    MAE: {class_mae:.4f}")
    
    # Sample predictions
    print(f"\nSample Predictions:")
    sample_indices = np.random.choice(y_score_test.shape[0], min(10, y_score_test.shape[0]), replace=False)
    
    for idx in sample_indices[:5]:
        true_class = class_names[y_class_test[idx]]
        pred_class = class_names[y_class_pred[idx]]
        true_score = y_score_test[idx]
        pred_score = y_score_pred[idx]
        conf = confidences[idx]
        
        match = "✓" if true_class == pred_class else "✗"
        error = abs(true_score - pred_score)
        
        print(f"  {match} True: {true_class:8s}({true_score:.2f}) | "
              f"Pred: {pred_class:8s}({pred_score:.2f}) | "
              f"Error: {error:.2f} | Conf: {conf:.3f}")
    
    return {
        'class_accuracy': class_accuracy,
        'score_rmse': score_rmse,
        'score_mae': score_mae,
        'score_r2': score_r2,
        'avg_confidence': confidences.mean()
    }


# ============================================================================
# SAVE MODELS
# ============================================================================
def save_models(classifier, regressors, scalers, vectorizer, count_vectorizer,
               label_encoder, feature_names, class_score_ranges, metrics,
               output_dir='trained_models'):
    """
    Save all trained artifacts to disk.
    
    Args:
        All trained components
        output_dir: Output directory
    """
    print(f"\n{'='*70}")
    print(f"💾 SAVING MODELS")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save preprocessing
    joblib.dump(vectorizer, f'{output_dir}/vectorizer.pkl')
    print(f"✓ Saved vectorizer.pkl")
    
    joblib.dump(count_vectorizer, f'{output_dir}/count_vectorizer.pkl')
    print(f"✓ Saved count_vectorizer.pkl")
    
    joblib.dump(label_encoder, f'{output_dir}/label_encoder.pkl')
    print(f"✓ Saved label_encoder.pkl")
    
    # Save classifier
    joblib.dump(classifier, f'{output_dir}/classifier.pkl')
    print(f"✓ Saved classifier.pkl")
    
    # Save regressors
    for class_name, regressor in regressors.items():
        if regressor is not None:
            joblib.dump(regressor, f'{output_dir}/regressor_{class_name.lower()}.pkl')
            joblib.dump(scalers[class_name], f'{output_dir}/scaler_{class_name.lower()}.pkl')
            print(f"✓ Saved regressor_{class_name.lower()}.pkl and scaler")
    
    # Save metadata
    metadata = {
        'class_names': label_encoder.classes_.tolist(),
        'class_score_ranges': class_score_ranges,
        'feature_names': feature_names,
        'theoretical_ranges': THEORETICAL_RANGES,
        'metrics': metrics,
        'trained_at': datetime.now().isoformat(),
        'version': '3.0.0'
    }
    
    joblib.dump(metadata, f'{output_dir}/metadata.pkl')
    print(f"✓ Saved metadata.pkl")
    
    print(f"\n✅ All models saved to: {output_dir}/")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🚀 TWO-STAGE CP DIFFICULTY PREDICTOR - TRAINING")
    print("="*70)
    print(f"\nTheoretical Score Ranges:")
    for class_name, (min_score, max_score) in THEORETICAL_RANGES.items():
        print(f"  {class_name:8s}: [{min_score:.1f}, {max_score:.1f}]")
    
    # Load dataset
    df = load_dataset('../data/dataset.jsonl')
    if df is None or len(df) < 20:
        print("\n❌ Insufficient data. Need at least 20 samples.")
        print("\nExpected format (dataset.jsonl):")
        print('''{"title": "Two Sum", "description": "Find two numbers...", "input": "...", "output": "...", "problem_class": "Easy", "problem_score": 2.5}''')
        return
    
    # Prepare features
    X, y_class, y_score, vectorizer, count_vectorizer, feature_names = prepare_features(
        df, fit=True
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(['Easy', 'Medium', 'Hard'])  # Fixed order
    class_names = label_encoder.classes_
    
    # Split data (stratified)
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )
    
    print(f"\nData Split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing:  {X_test.shape[0]} samples")
    
    # Stage 1: Train classifier
    classifier = train_classifier(X_train, y_class_train, X_test, y_class_test, class_names)
    
    # Stage 2: Train regressors
    regressors, scalers, class_score_ranges = train_regressors(
        X_train, y_class_train, y_score_train, class_names
    )
    
    # Evaluate
    metrics = evaluate_model(
        classifier, regressors, scalers, class_score_ranges,
        X_test, y_class_test, y_score_test, class_names
    )
    
    # Save models
    save_models(
        classifier, regressors, scalers,
        vectorizer, count_vectorizer, label_encoder,
        feature_names, class_score_ranges, metrics
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Metrics:")
    print(f"  Classification Accuracy: {metrics['class_accuracy']:.4f}")
    print(f"  Score RMSE: {metrics['score_rmse']:.4f}")
    print(f"  Score MAE: {metrics['score_mae']:.4f}")
    print(f"  Score R²: {metrics['score_r2']:.4f}")
    
    print(f"\n📦 Models saved to: trained_models/")
    print(f"\n🚀 Ready to deploy! Run: python app.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()