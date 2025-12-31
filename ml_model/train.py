import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# TextProcessor Class (MUST MATCH Flask App)
# ==============================================

class TextProcessor:
    def __init__(self):
        # Algorithm & Data Structure Keywords with weights (SAME as Flask)
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
        
        # Math Symbols & Operators (SAME as Flask)
        self.math_symbols = {
            'basic': {'+', '-', '*', '/', '=', '(', ')', '[', ']', '{', '}'},
            'comparison': {'<', '>', '<=', '>=', '==', '!=', '≡', '≈', '∼'},
            'advanced': {'∑', '∏', '∫', '∂', '∇', '∞', '√', '^', '**', '!', 'mod', '%'},
        }
        
        # Edge Case Indicators (SAME as Flask)
        self.edge_indicators = {
            'edge case', 'corner case', 'special case', 'consider',
            'note that', 'important', 'carefully', 'attention',
        }
        
    def preprocess_text(self, text):
        """Clean and preprocess text - EXACTLY as Flask app"""
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
        """Extract comprehensive features from text - EXACTLY as Flask app"""
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
            difficulty_words = {
                'easy': 0.5, 'simple': 0.5, 'straightforward': 0.5,
                'medium': 1.5, 'moderate': 1.5,
                'challenging': 2.5, 'difficult': 2.5, 'hard': 2.5,
                'very hard': 3.0, 'extremely hard': 3.5,
                'advanced': 2.5, 'expert': 3.0,
            }
            
            title_difficulty = 0
            for word, weight in difficulty_words.items():
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
        """Extract exactly 24 features - EXACTLY as Flask app"""
        features = self.extract_advanced_features(text, title)
        
        # IMPORTANT: This EXACT list of 24 features (SAME as Flask)
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

# ==============================================
# Main Training Function
# ==============================================

def train_models():
    print("Starting AutoJudge Model Training...")
    print("="*50)
    
    # Initialize processor (SAME as Flask app)
    processor = TextProcessor()
    
    # Load data
    print("1. Loading data...")
    try:
        data = []
        with open('../data/dataset.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} samples")
    except FileNotFoundError:
        print("Error: dataset.jsonl not found. Creating sample data...")
        df = create_sample_data()
    
    # Prepare features
    print("2. Extracting features...")
    tfidf_texts = []
    numeric_features_list = []
    
    for idx, row in df.iterrows():
        # Combine text (EXACTLY as Flask app does)
        combined_text = f"{row.get('title', '')} {row.get('description', '')} " \
                       f"{row.get('input', '')} {row.get('output', '')}"
        
        # Preprocess text (EXACTLY as Flask app)
        processed_text = processor.preprocess_text(combined_text)
        tfidf_texts.append(processed_text)
        
        # Extract numeric features (EXACTLY as Flask app)
        feature_vector, _ = processor.extract_feature_vector(combined_text, row.get('title', ''))
        numeric_features_list.append(feature_vector.flatten())
    
    # Create TF-IDF features
    print("3. Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(tfidf_texts)
    
    # Create numeric features
    print("4. Creating numeric features...")
    X_numeric = np.array(numeric_features_list)
    
    # VERIFY: We should have 24 features
    expected_features = 24
    if X_numeric.shape[1] != expected_features:
        print(f"❌ ERROR: Expected {expected_features} features, got {X_numeric.shape[1]}")
        print("This will cause errors in Flask app!")
        print("Feature names being extracted:")
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
        print(f"Expected: {feature_names}")
        return
    
    print(f"✅ Numeric features shape: {X_numeric.shape} (24 features as expected)")
    
    # Scale numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    
    # Combine features
    X = hstack([X_tfidf, X_numeric_scaled])
    print(f"✅ Total features shape: {X.shape} (TF-IDF: 5000 + Numeric: 24)")
    
    # Prepare targets
    print("5. Preparing targets...")
    
    if 'difficulty' in df.columns:
        label_encoder = LabelEncoder()
        y_class = label_encoder.fit_transform(df['difficulty'])
        class_names = label_encoder.classes_
        print(f"Classes: {class_names}")
    else:
        # Create synthetic targets
        print("Warning: 'difficulty' column not found. Creating synthetic targets.")
        word_counts = [len(text.split()) for text in tfidf_texts]
        y_class = np.zeros(len(df))
        for i, word_count in enumerate(word_counts):
            if word_count < 100:
                y_class[i] = 0  # Easy
            elif word_count < 300:
                y_class[i] = 1  # Medium
            else:
                y_class[i] = 2  # Hard
        
        label_encoder = LabelEncoder()
        y_class = label_encoder.fit_transform(y_class)
        class_names = ['Easy', 'Medium', 'Hard']
    
    if 'score' in df.columns:
        y_score = df['score'].values
    else:
        # Create synthetic scores (0-10)
        print("Warning: 'score' column not found. Creating synthetic scores.")
        y_score = np.zeros(len(df))
        for i, cls in enumerate(y_class):
            if cls == 0:  # Easy
                y_score[i] = np.random.uniform(2, 4)
            elif cls == 1:  # Medium
                y_score[i] = np.random.uniform(4, 7)
            else:  # Hard
                y_score[i] = np.random.uniform(7, 10)
    
    # Split data
    print("6. Splitting data...")
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Train Classification Model
    print("\n7. Training Classification Model...")
    print("-"*30)
    
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_class_train)
    
    # Evaluate classifier
    y_class_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_class_test, y_class_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_class_test, y_class_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_class_test, y_class_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('../web_app/static/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Train Regression Model
    print("\n8. Training Regression Model...")
    print("-"*30)
    
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_score_train)
    
    # Evaluate regressor
    y_score_pred = regressor.predict(X_test)
    y_score_pred = np.clip(y_score_pred, 0, 10)
    
    mae = mean_absolute_error(y_score_test, y_score_pred)
    rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_score_test, y_score_pred, alpha=0.6)
    plt.plot([0, 10], [0, 10], 'r--', alpha=0.5)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title('Actual vs Predicted Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig('../web_app/static/regression_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Save models and artifacts
    print("\n9. Saving models and artifacts...")
    
    # Create directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Save all artifacts
    joblib.dump(classifier, 'saved_models/classifier.pkl')
    joblib.dump(regressor, 'saved_models/regressor.pkl')
    joblib.dump(vectorizer, 'saved_models/vectorizer.pkl')
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    joblib.dump(label_encoder, 'saved_models/label_encoder.pkl')
    
    # Save feature names for verification
    feature_names_info = {
        'numeric_features': [
            'word_count', 'sentence_count', 'avg_word_length',
            'unique_word_ratio', 'total_keywords', 'total_keyword_weight',
            'keyword_dp', 'keyword_graph', 'keyword_advanced',
            'total_math_symbols', 'math_advanced', 'number_count',
            'max_number', 'section_count', 'constraint_count',
            'bullet_points', 'complexity_mentions', 'edge_case_mentions',
            'example_blocks', 'has_code', 'algo_complexity_score',
            'math_complexity_score', 'structural_complexity_score',
            'text_complexity_score'
        ],
        'expected_count': 24,
        'tfidf_features': 5000
    }
    
    joblib.dump(feature_names_info, 'saved_models/feature_info.pkl')
    
    print("✅ Models saved successfully!")
    print(f"  - Classifier: saved_models/classifier.pkl")
    print(f"  - Regressor: saved_models/regressor.pkl")
    print(f"  - Vectorizer: saved_models/vectorizer.pkl")
    print(f"  - Scaler: saved_models/scaler.pkl (expects 24 features)")
    print(f"  - Label Encoder: saved_models/label_encoder.pkl")
    
    # Verify scaler expectations
    print(f"\n🔍 Scaler verification:")
    print(f"   Scaler expects: {scaler.n_features_in_} features")
    print(f"   Flask app will provide: 24 features")
    
    if scaler.n_features_in_ == 24:
        print("   ✅ PERFECT MATCH!")
    else:
        print(f"   ❌ MISMATCH! This will cause errors in Flask app.")
    
    # Test with a sample
    print("\n10. Testing with sample text...")
    sample_text = """
    Given an array of integers, find the maximum sum subarray using Kadane's algorithm.
    This is a dynamic programming problem that requires O(n) time complexity.
    
    Input:
    The first line contains an integer n (1 ≤ n ≤ 10^5).
    The second line contains n space-separated integers.
    
    Output:
    Print the maximum sum.
    
    Example:
    Input: 5
           1 -2 3 4 -5
    Output: 7
    """
    
    sample_title = "Maximum Subarray Sum"
    
    # Process exactly as Flask app would
    combined = f"{sample_title} {sample_text}"
    processed = processor.preprocess_text(combined)
    feature_vector, features = processor.extract_feature_vector(combined, sample_title)
    
    print(f"Sample processing complete.")
    print(f"Extracted features: {feature_vector.shape[1]}")
    print(f"Word count: {features.get('word_count', 0)}")
    print(f"Total keywords: {features.get('total_keywords', 0)}")
    print(f"Has DP: {bool(features.get('has_dp', 0))}")
    
    if feature_vector.shape[1] == 24:
        print("✅ Feature extraction matches Flask app perfectly!")
    else:
        print(f"❌ Feature count mismatch!")
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

def create_sample_data():
    """Create sample data if dataset.jsonl doesn't exist"""
    sample_data = []
    
    # Easy problems
    easy_titles = ["Two Sum", "Palindrome Check", "Reverse String", "FizzBuzz", "Find Maximum"]
    for i, title in enumerate(easy_titles):
        sample_data.append({
            "title": title,
            "description": f"Simple {title.lower()} problem for beginners with basic implementation.",
            "input": "Standard input format",
            "output": "Single integer or string output",
            "difficulty": "Easy",
            "score": round(np.random.uniform(2.0, 4.0), 1)
        })
    
    # Medium problems
    medium_titles = ["Binary Search", "Merge Sort", "Tree Traversal", "Graph BFS", "Dynamic Programming Basics"]
    for i, title in enumerate(medium_titles):
        sample_data.append({
            "title": title,
            "description": f"Medium difficulty {title.lower()} requiring algorithmic thinking and optimization.",
            "input": "Multiple test cases with constraints",
            "output": "Complex output format",
            "difficulty": "Medium",
            "score": round(np.random.uniform(4.0, 7.0), 1)
        })
    
    # Hard problems
    hard_titles = ["Traveling Salesman", "Maximum Flow", "Segment Tree", "FFT Multiplication", "Convex Hull"]
    for i, title in enumerate(hard_titles):
        sample_data.append({
            "title": title,
            "description": f"Advanced {title.lower()} problem requiring sophisticated algorithms and data structures. Time complexity optimization needed.",
            "input": "Large constraints up to 10^6, multiple parameters",
            "output": "Multiple values or complex structure",
            "difficulty": "Hard",
            "score": round(np.random.uniform(7.0, 10.0), 1)
        })
    
    # Add more samples
    for i in range(90):
        difficulty = np.random.choice(['Easy', 'Medium', 'Hard'], p=[0.4, 0.4, 0.2])
        if difficulty == 'Easy':
            title = f"Problem {i+10}"
            score = np.random.uniform(2, 4)
        elif difficulty == 'Medium':
            title = f"Algorithm {i+10}"
            score = np.random.uniform(4, 7)
        else:
            title = f"Advanced {i+10}"
            score = np.random.uniform(7, 10)
        
        sample_data.append({
            "title": title,
            "description": f"Sample {difficulty.lower()} difficulty problem with various algorithmic concepts.",
            "input": "Sample input description",
            "output": "Sample output description",
            "difficulty": difficulty,
            "score": round(score, 1)
        })
    
    return pd.DataFrame(sample_data)

if __name__ == "__main__":
    train_models()