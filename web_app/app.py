"""
Production Flask API for Two-Stage CP Difficulty Predictor

ARCHITECTURE:
    STAGE 1: Classify into Easy/Medium/Hard with probabilities
    STAGE 2: Per-class regressor predicts score within range
    
CRITICAL REQUIREMENTS:
    1. Preprocessing MUST match training exactly
    2. Feature extraction MUST match training exactly
    3. Feature order MUST be preserved
    4. NO data leakage (never use class/score as input)
    
Author: Senior ML Engineer
Version: 3.0.0 - Production Ready
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import re
import os
import traceback
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# FLASK APP SETUP
# ============================================================================
app = Flask(__name__, 
            template_folder='../web_app/templates',
            static_folder='../web_app/static')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


# ============================================================================
# GLOBAL MODEL ARTIFACTS
# ============================================================================
class ModelArtifacts:
    """Container for all trained model artifacts"""
    def __init__(self):
        # Preprocessing
        self.vectorizer = None
        self.count_vectorizer = None
        self.label_encoder = None
        
        # Models
        self.classifier = None
        self.regressors = {}
        self.scalers = {}
        
        # Metadata
        self.class_names = []
        self.class_score_ranges = {}
        self.feature_names = []
        self.theoretical_ranges = {}
        
        # Status
        self.is_loaded = False
        self.load_timestamp = None

# Global instance
artifacts = ModelArtifacts()


# ============================================================================
# THEORETICAL SCORE RANGES (FLEXIBLE - MATCHES TRAINING)
# ============================================================================
THEORETICAL_RANGES = {
    'Easy': (0.0, 5.0),
    'Medium': (4.0, 8.0),
    'Hard': (6.0, 10.0)
}


# ============================================================================
# TEXT PREPROCESSING - MUST BE IDENTICAL TO TRAINING
# ============================================================================
def clean_text(text):
    """
    Clean and normalize text.
    MUST be 100% identical to train.py version.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Preserve mathematical operators
    text = re.sub(r'(\d+)\s*[+\-*/=<>]\s*(\d+)', r'\1 \2', text)
    
    # Keep alphanumeric + important symbols
    text = re.sub(r'[^\w\s\+\-\*/=<>\(\)\[\]\{\}\.,;:!?\^&\|%#@$\n_]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_engineered_features(text):
    """
    Extract hand-crafted features.
    MUST be 100% identical to train.py version.
    
    Args:
        text: Cleaned text string
        
    Returns:
        Dictionary of feature_name -> numeric_value
    """
    features = {}
    words = text.split()
    
    # 1. TEXT STATISTICS (4 features)
    features['text_length'] = len(text)
    features['word_count'] = len(words)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0.0
    features['char_diversity'] = len(set(text)) / max(1, len(text))
    
    # 2. MATHEMATICAL OPERATIONS (9 features)
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
    
    # 3. ALGORITHM CONCEPTS (weighted, 27 features)
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
    
    # 4. NUMERICAL CONTENT (2 features)
    features['number_count'] = len(re.findall(r'\b\d+\b', text))
    features['number_ratio'] = features['number_count'] / max(1, features['word_count'])
    
    # 5. STRUCTURE INDICATORS (6 features)
    features['sentence_count'] = len(re.split(r'[.!?]+', text))
    features['has_constraints'] = int(bool(re.search(r'\d+\s*[≤<>=]\s*\d+', text)))
    features['has_formula'] = int(bool(re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=', text)))
    features['has_loop'] = int('for' in text or 'while' in text or 'loop' in text)
    features['has_recursion'] = int('recursion' in text or 'recursive' in text)
    features['has_dp'] = int('dp' in text or 'dynamic' in text or 'memoization' in text)
    
    # 6. COMPLEXITY KEYWORDS (1 feature)
    complexity_keywords = ['complexity', 'optimize', 'efficient', 'time limit', 'space']
    features['complexity_keywords'] = sum(1 for kw in complexity_keywords if kw in text)
    
    # 7. CODE PATTERNS (1 feature)
    features['code_like'] = int(bool(re.search(r'[{}();=]', text)))
    
    return features


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_models(model_dir='trained_models'):
    """
    Load all trained model artifacts.
    
    Args:
        model_dir: Directory containing .pkl files
        
    Returns:
        bool: True if successful, False otherwise
    """
    global artifacts
    
    print(f"\n{'='*70}")
    print(f"🔍 LOADING MODELS FROM {model_dir}")
    print(f"{'='*70}")
    
    # Try multiple possible paths
    possible_paths = [
        model_dir,
        'trained_models',
        'enhanced_models',
        '../ml_model/trained_models',
        '../ml_model/enhanced_models',
        './ml_model/trained_models',
        './ml_model/enhanced_models',
        os.path.join(os.path.dirname(__file__), 'trained_models'),
        os.path.join(os.path.dirname(__file__), 'enhanced_models'),
        os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'trained_models'),
        os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'enhanced_models')
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            model_path = path
            print(f"✓ Found model directory: {os.path.abspath(path)}")
            break
    
    if not model_path:
        print("❌ Model directory not found!")
        print("   Searched in:")
        for path in possible_paths:
            print(f"   - {os.path.abspath(path)}")
        print("\n   Please train models first: python train.py")
        return False
    
    try:
        # Load preprocessing
        print("\n📦 Loading preprocessing artifacts...")
        artifacts.vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.pkl'))
        artifacts.count_vectorizer = joblib.load(os.path.join(model_path, 'count_vectorizer.pkl'))
        artifacts.label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
        artifacts.class_names = list(artifacts.label_encoder.classes_)
        print(f"✓ Vectorizers and encoder loaded")
        print(f"  Classes: {artifacts.class_names}")
        
        # Load metadata
        print("\n📋 Loading metadata...")
        metadata = joblib.load(os.path.join(model_path, 'metadata.pkl'))
        artifacts.class_score_ranges = metadata['class_score_ranges']
        artifacts.feature_names = metadata['feature_names']
        artifacts.theoretical_ranges = metadata.get('theoretical_ranges', THEORETICAL_RANGES)
        print(f"✓ Metadata loaded")
        print(f"  Feature names: {len(artifacts.feature_names)}")
        print(f"  Class ranges: {artifacts.class_score_ranges}")
        
        # Load classifier
        print("\n🎯 Loading STAGE 1 (Classifier)...")
        artifacts.classifier = joblib.load(os.path.join(model_path, 'classifier.pkl'))
        print(f"✓ Classifier: {artifacts.classifier.__class__.__name__}")
        
        # Verify predict_proba exists
        if not hasattr(artifacts.classifier, 'predict_proba'):
            print("⚠️  WARNING: Classifier missing predict_proba!")
            return False
        
        # Load regressors
        print("\n📊 Loading STAGE 2 (Regressors)...")
        for class_name in artifacts.class_names:
            regressor_path = os.path.join(model_path, f'regressor_{class_name.lower()}.pkl')
            scaler_path = os.path.join(model_path, f'scaler_{class_name.lower()}.pkl')
            
            if os.path.exists(regressor_path) and os.path.exists(scaler_path):
                artifacts.regressors[class_name] = joblib.load(regressor_path)
                artifacts.scalers[class_name] = joblib.load(scaler_path)
                print(f"✓ {class_name}: Loaded")
            else:
                artifacts.regressors[class_name] = None
                artifacts.scalers[class_name] = None
                print(f"⚠️  {class_name}: Not found (will use fallback)")
        
        artifacts.is_loaded = True
        artifacts.load_timestamp = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"✅ ALL MODELS LOADED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Classes: {artifacts.class_names}")
        print(f"Regressors: {[k for k, v in artifacts.regressors.items() if v is not None]}")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR loading models: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# PREDICTION PIPELINE
# ============================================================================
def predict_difficulty(title, description, input_desc="", output_desc=""):
    """
    Two-stage prediction pipeline.
    
    STAGE 1: Classify → Easy/Medium/Hard + probabilities
    STAGE 2: Regress → Score within class range
    
    Args:
        title: Problem title
        description: Problem description
        input_desc: Input description (optional)
        output_desc: Output description (optional)
        
    Returns:
        dict: Prediction results
    """
    if not artifacts.is_loaded:
        raise RuntimeError("Models not loaded")
    
    # ========================================
    # STEP 1: COMBINE AND CLEAN TEXT
    # ========================================
    combined_text = f"{title} {description} {input_desc} {output_desc}"
    cleaned_text = clean_text(combined_text)
    
    # ========================================
    # STEP 2: EXTRACT FEATURES
    # ========================================
    # Engineered features
    engineered_dict = extract_engineered_features(cleaned_text)
    engineered_df = pd.DataFrame([engineered_dict])
    
    # Fill missing features with 0
    for fname in artifacts.feature_names:
        if fname not in engineered_df.columns:
            engineered_df[fname] = 0.0
    
    # CRITICAL: Reorder to match training
    engineered_df = engineered_df[artifacts.feature_names]
    
    # TF-IDF features
    tfidf_features = artifacts.vectorizer.transform([cleaned_text])
    
    # Count features
    count_features = artifacts.count_vectorizer.transform([cleaned_text])
    
    # Combine: [TF-IDF | Count | Engineered]
    engineered_sparse = csr_matrix(engineered_df.values)
    X = hstack([tfidf_features, count_features, engineered_sparse])
    
    # ========================================
    # STAGE 1: CLASSIFY
    # ========================================
    class_probs = artifacts.classifier.predict_proba(X)[0]
    predicted_class_idx = int(np.argmax(class_probs))
    predicted_class = artifacts.class_names[predicted_class_idx]
    confidence = float(class_probs[predicted_class_idx])
    
    # Build probability dict
    class_prob_dict = {
        artifacts.class_names[i]: round(float(class_probs[i]), 4)
        for i in range(len(artifacts.class_names))
    }
    
    # ========================================
    # STAGE 2: PREDICT SCORE
    # ========================================
    # Get theoretical range
    theo_min, theo_max = artifacts.theoretical_ranges.get(
        predicted_class,
        THEORETICAL_RANGES[predicted_class]
    )
    
    # Predict score using class-specific regressor
    if (predicted_class in artifacts.regressors and 
        artifacts.regressors[predicted_class] is not None):
        try:
            # Predict normalized score [0, 1]
            regressor = artifacts.regressors[predicted_class]
            normalized_score = regressor.predict(X)[0]
            normalized_score = float(np.clip(normalized_score, 0.0, 1.0))
            
            # Denormalize using actual training range
            if predicted_class in artifacts.class_score_ranges:
                actual_min, actual_max = artifacts.class_score_ranges[predicted_class]
                denormalized_score = normalized_score * (actual_max - actual_min) + actual_min
            else:
                # Fallback to theoretical range
                denormalized_score = normalized_score * (theo_max - theo_min) + theo_min
            
            # Clip to theoretical range
            final_score = float(np.clip(denormalized_score, theo_min, theo_max))
            
        except Exception as e:
            print(f"⚠️  Regressor prediction failed: {e}")
            # Fallback to midpoint
            final_score = float((theo_min + theo_max) / 2.0)
            confidence *= 0.8
    else:
        # No regressor - use midpoint
        final_score = float((theo_min + theo_max) / 2.0)
        confidence *= 0.7
    
    # ========================================
    # BUILD RESPONSE
    # ========================================
    response = {
        "problem_class": predicted_class,
        "problem_score": round(final_score, 2),
        "confidence": round(confidence, 4),
        "class_probabilities": class_prob_dict,
        "theoretical_range": [float(theo_min), float(theo_max)],
        "metadata": {
            "text_length": len(combined_text),
            "word_count": len(combined_text.split()),
            "features_used": X.shape[1],
            "regressor_used": artifacts.regressors[predicted_class] is not None
        }
    }
    
    return response


# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/')
def index():
    """Render the main web interface"""
    return render_template('index.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        'service': 'CP Difficulty Predictor',
        'version': '3.0.0',
        'status': 'healthy' if artifacts.is_loaded else 'not ready',
        'models_loaded': artifacts.is_loaded,
        'endpoints': {
            'GET /': 'Web interface',
            'GET /api': 'API information',
            'POST /predict': 'Make prediction',
            'POST /api/predict': 'Make prediction (alias)',
            'GET /health': 'Health check',
            'GET /info': 'Model information'
        }
    })


@app.route('/health')
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy' if artifacts.is_loaded else 'unhealthy',
        'models_loaded': artifacts.is_loaded,
        'load_timestamp': artifacts.load_timestamp.isoformat() if artifacts.load_timestamp else None,
        'classes': artifacts.class_names,
        'regressors_available': [k for k, v in artifacts.regressors.items() if v is not None],
        'feature_count': len(artifacts.feature_names) if artifacts.feature_names else 0
    })


@app.route('/info')
def info():
    """Model information"""
    if not artifacts.is_loaded:
        return jsonify({'error': 'Models not loaded'}), 503
    
    return jsonify({
        'version': '3.0.0',
        'architecture': 'Two-Stage Ensemble',
        'classes': artifacts.class_names,
        'theoretical_ranges': artifacts.theoretical_ranges,
        'actual_score_ranges': artifacts.class_score_ranges,
        'features': {
            'total': len(artifacts.feature_names),
            'tfidf': artifacts.vectorizer.max_features,
            'count': artifacts.count_vectorizer.max_features,
            'engineered': len(artifacts.feature_names)
        },
        'models': {
            'classifier': artifacts.classifier.__class__.__name__,
            'regressors': {
                k: v.__class__.__name__ if v is not None else 'None'
                for k, v in artifacts.regressors.items()
            }
        }
    })


@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """
    Main prediction endpoint.
    
    Request JSON:
    {
        "title": "Problem title",
        "description": "Problem description",
        "input": "Input description (optional)",
        "output": "Output description (optional)"
    }
    
    Response JSON:
    {
        "problem_class": "Hard",
        "problem_score": 7.84,
        "confidence": 0.9723,
        "class_probabilities": {
            "Easy": 0.0123,
            "Medium": 0.0154,
            "Hard": 0.9723
        },
        "theoretical_range": [7.0, 10.0],
        "metadata": {
            "text_length": 456,
            "word_count": 78,
            "features_used": 3050,
            "regressor_used": true
        }
    }
    """
    try:
        # Check if models loaded
        if not artifacts.is_loaded:
            return jsonify({
                'error': 'Models not loaded',
                'message': 'Please restart server or train models first'
            }), 503
        
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract fields
        title = data.get('title', '').strip()
        description = data.get('description', '').strip()
        input_desc = data.get('input', '').strip()
        output_desc = data.get('output', '').strip()
        
        # Validate
        if not title:
            return jsonify({'error': 'Title is required'}), 400
        
        if not description:
            return jsonify({'error': 'Description is required'}), 400
        
        if len(description) < 20:
            return jsonify({'error': 'Description too short (min 20 characters)'}), 400
        
        # Make prediction
        result = predict_difficulty(title, description, input_desc, output_desc)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"\n❌ ERROR in prediction:")
        traceback.print_exc()
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if not artifacts.is_loaded:
            return jsonify({'error': 'Models not loaded'}), 503
        
        data = request.get_json()
        problems = data.get('problems', [])
        
        if not problems or not isinstance(problems, list):
            return jsonify({'error': 'Invalid request format'}), 400
        
        if len(problems) > 100:
            return jsonify({'error': 'Maximum 100 problems per batch'}), 400
        
        results = []
        for problem in problems:
            try:
                result = predict_difficulty(
                    problem.get('title', ''),
                    problem.get('description', ''),
                    problem.get('input', ''),
                    problem.get('output', '')
                )
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """404 handler"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """500 handler"""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    import sys
    
    print("\n" + "="*70)
    print("🚀 CP DIFFICULTY PREDICTOR - FLASK SERVER")
    print("="*70)
    print("\nArchitecture:")
    print("  STAGE 1: Classifier → Easy/Medium/Hard + probabilities")
    print("  STAGE 2: Per-class regressors → Scores within ranges")
    print("\nTheoretical Score Ranges:")
    for class_name, (min_s, max_s) in THEORETICAL_RANGES.items():
        print(f"  {class_name:8s}: [{min_s:.1f}, {max_s:.1f}]")
    print("="*70)
    
    # Allow specifying model directory from command line
    model_dir = 'trained_models'
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        print(f"\n📂 Using model directory from argument: {model_dir}")
    
    # Load models
    success = load_models(model_dir)
    
    if not success:
        print("\n❌ Failed to load models")
        print("   Server will start but predictions will fail")
        print("\n💡 Usage: python app.py [model_directory]")
        print("   Example: python app.py ../ml_model/trained_models")
    
    # Start server
    print("\n🌐 Starting server on http://0.0.0.0:5000")
    print("\nEndpoints:")
    print("  GET  /          - Service info")
    print("  GET  /health    - Health check")
    print("  GET  /info      - Model info")
    print("  POST /predict   - Single prediction")
    print("  POST /batch     - Batch predictions")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)