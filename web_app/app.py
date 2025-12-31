from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack, csr_matrix
import re
import os
import sys
import json

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
        missing_files = []
        for file in model_files:
            file_path = os.path.join(models_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
                print(f"Warning: {file} not found at {file_path}")
        
        if missing_files:
            print(f"Missing files: {missing_files}")
            print("Please train models first: cd ml_model && python train.py")
            is_loaded = False
            return
        
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
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting steps:")
        print("1. Run the training script first: cd ml_model && python train.py")
        print("2. Check if models are saved in ml_model/saved_models/")
        print("3. Verify all .pkl files exist")
        is_loaded = False

# Load models on startup
load_models()

class TextProcessor:
    def __init__(self):
        # Algorithm & Data Structure Keywords with weights
        self.keywords = {
            # Advanced Algorithms (High weight)
            'dp': 3.0, 'dynamic programming': 3.5, 'bitmask': 3.0, 'bitmask dp': 3.5,
            'segment tree': 3.0, 'fenwick tree': 3.0, 'suffix array': 3.0,
            'max flow': 3.0, 'min cut': 3.0, 'matching': 3.0,
            'treap': 2.5, 'splay tree': 2.5, 'aho-corasick': 3.0,
            
            # Graph Algorithms (Medium-High weight)
            'graph': 2.5, 'dijkstra': 2.5, 'bellman-ford': 2.5, 'floyd-warshall': 2.5,
            'topological sort': 2.0, 'strongly connected': 2.5, 'articulation point': 2.5,
            'bridge': 2.5, 'minimum spanning': 2.0, 'kruskal': 2.0, 'prim': 2.0,
            'bipartite': 2.0, 'max matching': 2.5, 'network flow': 2.5,
            'euler path': 2.0, 'hamiltonian': 2.5, 'traveling salesman': 3.0,
            
            # Search Algorithms (Medium weight)
            'bfs': 1.5, 'dfs': 1.5, 'backtracking': 2.0, 'branch and bound': 2.5,
            'meet in the middle': 2.0, 'iterative deepening': 2.0,
            'a star': 2.0, 'bidirectional': 2.0,
            
            # Tree Algorithms (Medium weight)
            'tree': 1.5, 'binary tree': 1.5, 'binary search tree': 1.5,
            'avl': 2.0, 'red-black': 2.0, 'trie': 2.0, 'segment': 2.5,
            'lowest common ancestor': 2.0, 'tree dp': 2.5,
            
            # String Algorithms (Medium weight)
            'kmp': 2.0, 'rabin-karp': 2.0, 'z algorithm': 2.0,
            'manacher': 2.0, 'palindrome': 1.5, 'string matching': 1.5,
            
            # Mathematical Concepts (Variable weight)
            'gcd': 1.0, 'lcm': 1.0, 'prime': 1.5, 'sieve': 1.5,
            'modular': 1.5, 'combinatorics': 2.0, 'probability': 2.0,
            'matrix exponentiation': 2.5, 'fft': 3.0, 'ntt': 3.0,
            'number theory': 2.0, 'geometry': 2.0, 'calculus': 2.5,
            'linear algebra': 2.5, 'game theory': 2.0,
            
            # Data Structures (Low-Medium weight)
            'stack': 1.0, 'queue': 1.0, 'deque': 1.0, 'priority queue': 1.5,
            'heap': 1.5, 'hash': 1.0, 'hashmap': 1.0, 'hashset': 1.0,
            'linked list': 1.0, 'union find': 1.5, 'disjoint set': 1.5,
            'binary indexed tree': 2.0, 'sparse table': 2.0,
            
            # Optimization & Techniques
            'greedy': 1.5, 'two pointer': 1.0, 'sliding window': 1.0,
            'prefix sum': 1.0, 'difference array': 1.5, 'binary search': 1.5,
            'ternary search': 2.0, 'divide and conquer': 2.0,
            'memoization': 2.0, 'tabulation': 2.0,
            
            # Complexity indicators
            'time complexity': 2.0, 'space complexity': 1.5, 'constraints': 1.0,
            'optimize': 2.0, 'efficient': 2.0, 'optimal': 2.0,
            'minimize': 1.5, 'maximize': 1.5,
        }
        
        # Math Symbols & Operators
        self.math_symbols = {
            'basic': {'+', '-', '*', '/', '=', '(', ')', '[', ']', '{', '}'},
            'comparison': {'<', '>', '<=', '>=', '==', '!=', '≡', '≈', '∼'},
            'advanced': {'∑', '∏', '∫', '∂', '∇', '∞', '√', '^', '**', '!', 'mod', '%'},
        }
        
        # Problem Type Indicators
        self.problem_types = {
            'implementation': 1.0,
            'simulation': 1.5,
            'brute force': 1.5,
            'constructive': 2.0,
            'interactive': 2.0,
            'optimization': 2.5,
        }
        
        # Edge Case Indicators
        self.edge_indicators = {
            'edge case', 'corner case', 'special case', 'consider',
            'note that', 'important', 'carefully', 'attention',
        }
        
        # Difficulty Level Words
        self.difficulty_words = {
            'easy': 0.5, 'simple': 0.5, 'straightforward': 0.5,
            'medium': 1.5, 'moderate': 1.5,
            'challenging': 2.5, 'difficult': 2.5, 'hard': 2.5,
            'very hard': 3.0, 'extremely hard': 3.5,
            'advanced': 2.5, 'expert': 3.0,
        }
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Preserve mathematical symbols
        for category in self.math_symbols.values():
            for symbol in category:
                text = text.replace(symbol, f' {symbol} ')
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove special characters except preserved symbols
        text = re.sub(r'[^\w\s\+\-\*/^<>=!%\(\)\[\]\{\}∑∏∫∂∇∞√∼≈≡]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_advanced_features(self, text, title=""):
        """Extract comprehensive features from text"""
        text_lower = text.lower()
        title_lower = title.lower() if title else ""
        
        features = {}
        
        # 1. Basic Text Statistics
        words = text_lower.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = sum(len(w) for w in words) / max(1, len(words))
        features['avg_sentence_length'] = len(words) / max(1, len(sentences))
        features['unique_word_ratio'] = len(set(words)) / max(1, len(words))
        
        # 2. Keyword Features
        keyword_counts = {}
        total_keyword_weight = 0
        keyword_categories = {
            'advanced': 0, 'graph': 0, 'dp': 0, 'math': 0,
            'data_structure': 0, 'search': 0, 'string': 0,
        }
        
        for keyword, weight in self.keywords.items():
            pattern = rf'\b{re.escape(keyword)}\b'
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                keyword_counts[keyword] = count
                total_keyword_weight += weight * count
                
                # Categorize
                if any(k in keyword for k in ['dp', 'dynamic', 'bitmask']):
                    keyword_categories['dp'] += count
                elif any(k in keyword for k in ['graph', 'bfs', 'dfs', 'dijkstra']):
                    keyword_categories['graph'] += count
                elif any(k in keyword for k in ['tree', 'heap', 'stack', 'queue', 'linked']):
                    keyword_categories['data_structure'] += count
                elif any(k in keyword for k in ['math', 'prime', 'gcd', 'modular', 'calculus']):
                    keyword_categories['math'] += count
                elif any(k in keyword for k in ['string', 'kmp', 'palindrome']):
                    keyword_categories['string'] += count
                elif any(k in keyword for k in ['segment', 'fenwick', 'suffix', 'fft', 'flow']):
                    keyword_categories['advanced'] += count
        
        features['total_keywords'] = sum(keyword_counts.values())
        features['unique_keywords'] = len(keyword_counts)
        features['total_keyword_weight'] = total_keyword_weight
        features['avg_keyword_weight'] = total_keyword_weight / max(1, sum(keyword_counts.values()))
        features.update({f'keyword_{k}': v for k, v in keyword_categories.items()})
        features['has_dp'] = int(keyword_categories['dp'] > 0)
        features['has_graph'] = int(keyword_categories['graph'] > 0)
        features['has_advanced'] = int(keyword_categories['advanced'] > 0)
        
        # 3. Math Features
        math_counts = {}
        total_math = 0
        for category_name, symbols in self.math_symbols.items():
            count = sum(text.count(symbol) for symbol in symbols)
            math_counts[f'math_{category_name}'] = count
            total_math += count
        
        features.update(math_counts)
        features['total_math_symbols'] = total_math
        
        # Count numbers in text
        numbers = [int(match) for match in re.findall(r'\b\d+\b', text) if match.isdigit()]
        features['number_count'] = len(numbers)
        features['max_number'] = max(numbers) if numbers else 0
        features['has_large_numbers'] = int(any(n > 1000 for n in numbers))
        features['has_very_large_numbers'] = int(any(n > 1000000 for n in numbers))
        
        # 4. Structural Features
        lines = text.split('\n')
        sections = {'input': 0, 'output': 0, 'example': 0, 'note': 0, 'constraints': 0}
        
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('input'):
                sections['input'] += 1
            elif line_lower.startswith('output'):
                sections['output'] += 1
            elif line_lower.startswith(('example', 'sample')):
                sections['example'] += 1
            elif line_lower.startswith('note'):
                sections['note'] += 1
            elif 'constraint' in line_lower:
                sections['constraints'] += 1
        
        features.update({f'has_{k}_section': int(v > 0) for k, v in sections.items()})
        features['section_count'] = sum(1 for v in sections.values() if v > 0)
        
        # Count bullet points and numbered lists
        bullet_points = sum(1 for line in lines if line.strip().startswith(('•', '-', '*', '◦')))
        numbered_lists = sum(1 for line in lines if re.match(r'^\d+[\.\)]', line.strip()))
        features['bullet_points'] = bullet_points
        features['numbered_lists'] = numbered_lists
        
        # 5. Complexity Features
        complexity_count = sum(text_lower.count(word) for word in 
                              ['time complexity', 'space complexity', 'optimize', 'efficient', 'optimal'])
        features['complexity_mentions'] = complexity_count
        
        edge_case_count = sum(text_lower.count(word) for word in self.edge_indicators)
        features['edge_case_mentions'] = edge_case_count
        
        # 6. Title Analysis
        if title:
            title_words = title_lower.split()
            features['title_word_count'] = len(title_words)
            
            # Check for difficulty in title
            title_difficulty = 0
            for word, weight in self.difficulty_words.items():
                if word in title_lower:
                    title_difficulty = max(title_difficulty, weight)
            features['title_difficulty_score'] = title_difficulty
            features['title_has_difficulty'] = int(title_difficulty > 0)
            
            # Check for algorithms in title
            title_has_algorithm = any(keyword in title_lower for keyword in self.keywords.keys())
            features['title_has_algorithm'] = int(title_has_algorithm)
        
        # 7. Constraint Analysis
        constraint_matches = re.findall(r'(\d+)\s*[≤<]\s*[A-Za-z]', text)
        constraint_matches += re.findall(r'[A-Za-z]\s*[≤<]\s*(\d+)', text)
        constraints = [int(m) for m in constraint_matches if m.isdigit()]
        
        features['constraint_count'] = len(constraints)
        features['max_constraint'] = max(constraints) if constraints else 0
        
        # 8. Example Analysis
        example_blocks = text_lower.count('example') + text_lower.count('sample')
        features['example_blocks'] = example_blocks
        features['has_examples'] = int(example_blocks > 0)
        
        # Count input/output pairs
        io_pairs = text_lower.count('input:') + text_lower.count('output:')
        features['io_pairs'] = io_pairs
        
        # 9. Code Features
        has_code = any(indicator in text for indicator in 
                      ['def ', 'class ', 'int ', 'void ', 'function ', 'public ', 'private '])
        features['has_code'] = int(has_code)
        
        # 10. Composite Scores
        # Algorithm Complexity Score
        algo_score = (
            features.get('keyword_advanced', 0) * 3 +
            features.get('keyword_dp', 0) * 2.5 +
            features.get('keyword_graph', 0) * 2 +
            features.get('total_keyword_weight', 0) * 0.1
        )
        
        # Mathematical Complexity Score
        math_score = (
            features.get('total_math_symbols', 0) * 0.5 +
            features.get('math_advanced', 0) * 2 +
            features.get('number_count', 0) * 0.1 +
            features.get('max_number', 0) * 0.001
        )
        
        # Structural Complexity Score
        structural_score = (
            features.get('section_count', 0) * 0.5 +
            features.get('bullet_points', 0) * 0.2 +
            features.get('numbered_lists', 0) * 0.2 +
            features.get('constraint_count', 0) * 0.8
        )
        
        # Text Complexity Score
        text_score = (
            min(features.get('word_count', 0) / 100, 3) +
            min(features.get('sentence_count', 0) / 10, 2) +
            features.get('unique_word_ratio', 0) * 2
        )
        
        features['algo_complexity_score'] = algo_score
        features['math_complexity_score'] = math_score
        features['structural_complexity_score'] = structural_score
        features['text_complexity_score'] = text_score
        features['total_composite_score'] = algo_score + math_score + structural_score + text_score
        
        return features
    
    def extract_feature_vector(self, text, title=""):
        """Extract features as numpy array for model input"""
        features = self.extract_advanced_features(text, title)
        
        # Select important features for the model
        feature_names = [
            'word_count', 'sentence_count', 'avg_word_length',
            'unique_word_ratio', 'total_keywords', 'total_keyword_weight',
            'keyword_dp', 'keyword_graph', 'keyword_advanced',
            'total_math_symbols', 'math_advanced', 'number_count',
            'max_number', 'section_count', 'constraint_count',
            'bullet_points', 'complexity_mentions', 'edge_case_mentions',
            'example_blocks', 'has_code', 'algo_complexity_score',
            'math_complexity_score', 'structural_complexity_score',
            'text_complexity_score'
        ]
        
        # Create feature vector
        feature_vector = []
        for name in feature_names:
            feature_vector.append(features.get(name, 0))
        
        return np.array(feature_vector).reshape(1, -1), features

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
        
        # Combine text for TF-IDF
        combined_text = f"{title} {description} {input_desc} {output_desc}"
        
        # Preprocess text
        processed_text = processor.preprocess_text(combined_text)
        
        # Create features for ML model
        tfidf_features = vectorizer.transform([processed_text])
        
        # Extract advanced features
        feature_vector, detailed_features = processor.extract_feature_vector(combined_text, title)
        
        # Scale numeric features
        numeric_scaled = scaler.transform(feature_vector)
        
        # Debug information
        print(f"\n{'='*50}")
        print("PREDICTION REQUEST")
        print(f"{'='*50}")
        print(f"Title: {title[:50]}...")
        print(f"Description length: {len(description)} chars")
        print(f"Processed text length: {len(processed_text)} chars")
        print(f"TF-IDF shape: {tfidf_features.shape}")
        print(f"Numeric features shape: {numeric_scaled.shape}")
        print(f"Total keywords: {detailed_features.get('total_keywords', 0)}")
        print(f"Has DP: {bool(detailed_features.get('has_dp', 0))}")
        print(f"Has Graph: {bool(detailed_features.get('has_graph', 0))}")
        
        # Combine features
        X = hstack([tfidf_features, numeric_scaled])
        
        # Make predictions
        class_pred = classifier.predict(X)[0]
        score_pred = regressor.predict(X)[0]
        
        # Clip score to 0-10
        score_pred = max(0.0, min(10.0, float(score_pred)))
        
        # Get class label
        class_label = label_encoder.inverse_transform([class_pred])[0]
        
        # Calculate difficulty class based on score (more accurate)
        def get_difficulty_class(score):
            if score <= 3.33:
                return 'Easy'
            elif score <= 6.67:
                return 'Medium'
            else:
                return 'Hard'
        
        score_based_class = get_difficulty_class(score_pred)
        
        # Calculate confidence scores
        if hasattr(classifier, 'predict_proba'):
            class_probs = classifier.predict_proba(X)[0]
            class_confidence = float(max(class_probs))
        else:
            class_confidence = 0.8
        
        # Calculate score confidence based on position in scale
        if 3.0 <= score_pred <= 7.0:
            # Middle range - higher confidence
            score_confidence = 0.85
        elif 1.0 <= score_pred < 3.0 or 7.0 < score_pred <= 9.0:
            score_confidence = 0.75
        else:
            # Extreme values
            score_confidence = 0.65
        
        # Adjust confidence based on feature consistency
        keyword_count = detailed_features.get('total_keywords', 0)
        if keyword_count > 5:
            class_confidence = min(0.95, class_confidence * 1.1)
        
        # Prepare detailed analysis for frontend
        feature_analysis = {
            'text_statistics': {
                'words': detailed_features.get('word_count', 0),
                'sentences': detailed_features.get('sentence_count', 0),
                'unique_words_ratio': round(detailed_features.get('unique_word_ratio', 0) * 100, 1)
            },
            'algorithms_detected': {
                'total': detailed_features.get('total_keywords', 0),
                'has_dp': bool(detailed_features.get('has_dp', 0)),
                'has_graph': bool(detailed_features.get('has_graph', 0)),
                'has_advanced': bool(detailed_features.get('has_advanced', 0))
            },
            'mathematical_complexity': {
                'math_symbols': detailed_features.get('total_math_symbols', 0),
                'numbers_found': detailed_features.get('number_count', 0),
                'large_numbers': bool(detailed_features.get('has_large_numbers', 0))
            },
            'structural_analysis': {
                'sections': detailed_features.get('section_count', 0),
                'constraints': detailed_features.get('constraint_count', 0),
                'examples': detailed_features.get('example_blocks', 0)
            }
        }
        
        # Generate insights based on features
        insights = []
        if detailed_features.get('has_dp', 0):
            insights.append("Dynamic programming detected - indicates medium to high difficulty")
        if detailed_features.get('has_graph', 0):
            insights.append("Graph algorithms present - requires good data structure knowledge")
        if detailed_features.get('total_math_symbols', 0) > 10:
            insights.append("High mathematical complexity detected")
        if detailed_features.get('constraint_count', 0) > 3:
            insights.append("Multiple constraints require careful handling")
        if detailed_features.get('example_blocks', 0) == 0:
            insights.append("No examples provided - might be less clear")
        
        # If no specific insights, add generic ones
        if not insights:
            if score_pred < 4:
                insights.append("Problem appears straightforward")
            elif score_pred < 7:
                insights.append("Moderate difficulty - requires algorithmic thinking")
            else:
                insights.append("High difficulty - advanced concepts required")
        
        response = {
            'success': True,
            'prediction': {
                'problem_class': class_label,
                'score_based_class': score_based_class,
                'class_confidence': round(class_confidence, 3),
                'problem_score': round(score_pred, 2),
                'score_confidence': round(score_confidence, 3),
                'detailed_features': detailed_features,
                'feature_analysis': feature_analysis,
                'insights': insights,
                'metadata': {
                    'model_version': '1.0',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'features_used': len(detailed_features)
                }
            }
        }
        
        print(f"Prediction: {score_based_class} (Score: {score_pred:.2f}/10)")
        print(f"Confidence: {class_confidence:.2%}")
        print(f"{'='*50}\n")
        
        return jsonify(response)
        
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
            'output_description': 'Print the indices of two numbers that sum to target.',
            'expected_difficulty': 'Easy'
        },
        {
            'title': 'Binary Tree Level Order Traversal',
            'description': 'Given the root of a binary tree, return the level order traversal of its nodes values. (i.e., from left to right, level by level).',
            'input_description': 'The input contains the tree nodes in level order format. Use -1 for null nodes.',
            'output_description': 'Print each level on a separate line.',
            'expected_difficulty': 'Medium'
        },
        {
            'title': 'Regular Expression Matching',
            'description': 'Given an input string s and a pattern p, implement regular expression matching with support for . and * where: . Matches any single character. * Matches zero or more of the preceding element. The matching should cover the entire input string (not partial).',
            'input_description': 'First line contains string s. Second line contains pattern p.',
            'output_description': "Print 'true' if pattern matches the entire string, otherwise 'false'.",
            'expected_difficulty': 'Hard'
        },
        {
            'title': 'Shortest Path in Weighted Graph (Dijkstra)',
            'description': 'Given a weighted directed graph with n nodes and m edges, find the shortest path from node 1 to node n using Dijkstra\'s algorithm. Edge weights are positive integers. Return the shortest distance or -1 if no path exists.',
            'input_description': 'First line contains n and m. Next m lines contain u v w representing an edge from u to v with weight w. Constraints: 1 ≤ n ≤ 10^5, 1 ≤ m ≤ 2 * 10^5, 1 ≤ w ≤ 10^9.',
            'output_description': 'Print the shortest distance, or -1 if no path exists.',
            'expected_difficulty': 'Medium'
        },
        {
            'title': 'Dynamic Programming: Coin Change',
            'description': 'Given an array of coin denominations and a target amount, return the minimum number of coins needed to make up that amount. If that amount cannot be made up, return -1. You may assume an infinite number of each kind of coin.',
            'input_description': 'First line contains n and amount. Second line contains n space-separated integers representing coin denominations.',
            'output_description': 'Print the minimum number of coins needed.',
            'expected_difficulty': 'Medium'
        }
    ]
    return jsonify({'samples': samples, 'count': len(samples)})

@app.route('/analyze', methods=['POST'])
def analyze_features():
    """Endpoint to get detailed feature analysis without prediction"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        title = data.get('title', '')
        description = data.get('description', '')
        
        if not description:
            return jsonify({'success': False, 'error': 'Description required'})
        
        combined_text = f"{title} {description}"
        _, detailed_features = processor.extract_feature_vector(combined_text, title)
        
        # Clean up features for JSON serialization
        cleaned_features = {}
        for key, value in detailed_features.items():
            if isinstance(value, (int, float, str, bool)):
                cleaned_features[key] = value
            elif isinstance(value, np.integer):
                cleaned_features[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned_features[key] = float(value)
            else:
                cleaned_features[key] = str(value)
        
        return jsonify({
            'success': True,
            'feature_analysis': cleaned_features
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 AutoJudge AI - Advanced Difficulty Predictor")
    print("="*60)
    print(f"📊 Models loaded: {'✅ READY' if is_loaded else '❌ NOT LOADED'}")
    print(f"🌐 Web Interface: http://localhost:5000")
    print(f"📈 Health Check: http://localhost:5000/health")
    print(f"🔍 Feature Analyzer: POST to http://localhost:5000/analyze")
    print("="*60)
    
    if not is_loaded:
        print("\n⚠️  WARNING: ML models not loaded!")
        print("Please train the models first:")
        print("  cd ml_model")
        print("  python train.py")
        print("\nThe web app will still run with demo functionality.")
        print("Sample predictions will be generated.")
    
    print("\n📋 Available endpoints:")
    print("  GET  /              - Web interface")
    print("  GET  /health        - System health check")
    print("  GET  /sample        - Sample problems")
    print("  POST /predict       - Make prediction")
    print("  POST /analyze       - Feature analysis only")
    print("="*60 + "\n")
    
    # Run the app
    app.run(debug=True, port=5000, host='0.0.0.0')