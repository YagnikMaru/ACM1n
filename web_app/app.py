from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack
import re
import os
import sys

# Add parent directory to path to import ml_model modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)

# Initialize with defaults
classifier = None
regressor = None
vectorizer = None
scaler = None
label_encoder = None
is_loaded = False

def load_models():
    """Load ML models and artifacts"""
    global classifier, regressor, vectorizer, scaler, label_encoder, is_loaded
    
    try:
        print("Loading models...")
        
        # Get absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, '..', 'ml_model', 'saved_models')
        
        print(f"Looking for models in: {models_dir}")
        
        # Check if models exist
        model_files = ['classifier.pkl', 'regressor.pkl', 'vectorizer.pkl', 'scaler.pkl', 'label_encoder.pkl']
        for file in model_files:
            file_path = os.path.join(models_dir, file)
            if not os.path.exists(file_path):
                print(f"Warning: {file} not found at {file_path}")
        
        # Load models
        classifier = joblib.load(os.path.join(models_dir, 'classifier.pkl'))
        print("✓ Classifier loaded")
        
        regressor = joblib.load(os.path.join(models_dir, 'regressor.pkl'))
        print("✓ Regressor loaded")
        
        vectorizer = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
        print("✓ Vectorizer loaded")
        
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        print("✓ Scaler loaded")
        
        label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        print("✓ Label encoder loaded")
        
        is_loaded = True
        print("✅ All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nTroubleshooting steps:")
        print("1. Run the training script first: cd ml_model && python train.py")
        print("2. Check if models are saved in ml_model/saved_models/")
        print("3. Verify all .pkl files exist")
        is_loaded = False

# Load models on startup
load_models()

class TextProcessor:
    def __init__(self):
        self.keywords = ['dp', 'graph', 'tree', 'greedy', 'math', 'recursion', 
                        'bitmask', 'bfs', 'dfs', 'binary search', 'dynamic programming',
                        'backtracking', 'dijkstra', 'sort', 'stack', 'queue']
        self.math_symbols = {'+', '-', '*', '/', '^', '<=', '>=', '==', '!=', '=', '<', '>'}
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Preserve mathematical symbols
        for symbol in self.math_symbols:
            text = text.replace(symbol, f' {symbol} ')
        
        # Remove special characters except preserved symbols
        text = re.sub(r'[^\w\s\+\-\*/^<>=]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text):
        """Extract text features"""
        features = []
        
        # Word count
        words = text.split()
        features.append(len(words))
        
        # Unique words
        features.append(len(set(words)))
        
        # Average word length
        if words:
            features.append(sum(len(w) for w in words) / len(words))
        else:
            features.append(0)
        
        # Count mathematical symbols
        math_count = 0
        for symbol in self.math_symbols:
            math_count += text.count(symbol)
        features.append(math_count)
        
        # Count programming keywords
        keyword_count = 0
        for keyword in self.keywords:
            # Use word boundaries for exact match
            pattern = rf'\b{re.escape(keyword)}\b'
            keyword_count += len(re.findall(pattern, text))
        features.append(keyword_count)
        
        # Sentence count (approximate)
        sentence_enders = ['.', '?', '!']
        features.append(sum(text.count(ender) for ender in sentence_enders))
        
        # Complexity indicators
        features.append(1 if any(word in text for word in ['recursion', 'recursive']) else 0)
        features.append(1 if any(word in text for word in ['dp', 'dynamic programming']) else 0)
        features.append(1 if any(word in text for word in ['graph', 'node', 'edge', 'vertex']) else 0)
        
        return features

processor = TextProcessor()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', models_loaded=is_loaded)

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if is_loaded else 'degraded',
        'models_loaded': is_loaded,
        'message': 'Models loaded successfully' if is_loaded else 'Models not loaded. Please train models first.'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Check if models are loaded
        if not is_loaded:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please train the models first.',
                'instructions': 'Run: cd ml_model && python train.py'
            })
        
        # Get data from request
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            })
        
        title = data.get('title', '')
        description = data.get('description', '')
        input_desc = data.get('input_description', '')
        output_desc = data.get('output_description', '')
        
        # Validate required fields
        if not title or not description:
            return jsonify({
                'success': False,
                'error': 'Title and description are required'
            })
        
        # Combine text
        combined_text = f"{title} {description} {input_desc} {output_desc}"
        
        # Preprocess text
        processed_text = processor.preprocess_text(combined_text)
        
        # Debug: Print processed text
        print(f"Processed text length: {len(processed_text)}")
        
        # Create features
        tfidf_features = vectorizer.transform([processed_text])
        numeric_features = processor.extract_features(processed_text)
        
        # Ensure numeric_features has correct shape
        numeric_features = np.array(numeric_features).reshape(1, -1)
        numeric_scaled = scaler.transform(numeric_features)
        
        # Debug: Print feature shapes
        print(f"TF-IDF shape: {tfidf_features.shape}")
        print(f"Numeric shape: {numeric_scaled.shape}")
        
        # Combine features
        X = hstack([tfidf_features, numeric_scaled])
        
        # Make predictions
        class_pred = classifier.predict(X)[0]
        score_pred = regressor.predict(X)[0]
        
        # Clip score to 0-10
        score_pred = max(0.0, min(10.0, float(score_pred)))
        
        # Get class label
        class_label = label_encoder.inverse_transform([class_pred])[0]
        
        # Get probability/confidence
        if hasattr(classifier, 'predict_proba'):
            confidence = float(classifier.predict_proba(X).max())
        else:
            confidence = 0.8  # Default confidence
        
        # Calculate score confidence based on position in scale
        if 3.0 <= score_pred <= 7.0:
            # Middle range - higher confidence
            score_confidence = 0.85
        elif 1.0 <= score_pred < 3.0 or 7.0 < score_pred <= 9.0:
            score_confidence = 0.75
        else:
            # Extreme values
            score_confidence = 0.65
        
        return jsonify({
            'success': True,
            'prediction': {
                'problem_class': class_label,
                'class_confidence': round(confidence, 3),
                'problem_score': round(score_pred, 2),
                'score_confidence': round(score_confidence, 3)
            }
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'details': 'Check console for full error trace'
        })

@app.route('/sample', methods=['GET'])
def get_sample():
    """Return sample problems"""
    samples = [
        {
            'title': 'Two Sum',
            'description': 'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order.',
            'input_description': 'First line contains an integer n, the size of array. Next line contains n space-separated integers. The last line contains the target sum.',
            'output_description': 'Print the indices of two numbers that sum to target.'
        },
        {
            'title': 'Binary Tree Level Order Traversal',
            'description': 'Given the root of a binary tree, return the level order traversal of its nodes values. (i.e., from left to right, level by level).',
            'input_description': 'The input contains the tree nodes in level order format. Use -1 for null nodes.',
            'output_description': 'Print each level on a separate line.'
        },
        {
            'title': 'Regular Expression Matching',
            'description': 'Given an input string s and a pattern p, implement regular expression matching with support for . and * where: . Matches any single character. * Matches zero or more of the preceding element. The matching should cover the entire input string (not partial).',
            'input_description': 'First line contains string s. Second line contains pattern p.',
            'output_description': "Print 'true' if pattern matches the entire string, otherwise 'false'."
        }
    ]
    return jsonify({'samples': samples, 'count': len(samples)})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("AutoJudge Web Application")
    print("="*50)
    print(f"Models loaded: {'✅' if is_loaded else '❌'}")
    print(f"Application running at: http://localhost:5000")
    print(f"Health check: http://localhost:5000/health")
    print("="*50 + "\n")
    
    if not is_loaded:
        print("⚠️  WARNING: Models not loaded!")
        print("Please run the training script first:")
        print("cd ml_model && python train.py")
        print("\nThe web app will still run, but predictions will fail.")
    
    app.run(debug=True, port=5000, host='0.0.0.0')