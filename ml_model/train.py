import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# DETERMINISM & REPRODUCIBILITY
# ==============================================
SEED = 42
np.random.seed(SEED)
import random
random.seed(SEED)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           mean_absolute_error, mean_squared_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb
from collections import Counter

# ==============================================
# ULTRA-ENHANCED TextProcessor Class
# ==============================================

class UltraEnhancedTextProcessor:
    """Ultra-enhanced text processor with 50+ features and deterministic preprocessing"""
    
    def __init__(self):
        # Set random seed for reproducibility
        np.random.seed(SEED)
        
        # Comprehensive Algorithm Dictionary with hierarchical weights
        self.algorithms = {
            # CORE ADVANCED ALGORITHMS (Highest weight)
            'dynamic_programming': {
                'keywords': ['dp', 'dynamic programming', 'memoization', 'tabulation', 
                           'knapsack', 'lcs', 'lis', 'bitmask dp', 'digit dp', 'tree dp',
                           'dp on trees', 'dp on graphs', 'state compression'],
                'weight': 4.0
            },
            'graph_advanced': {
                'keywords': ['max flow', 'min cut', 'dinic', 'edmonds karp', 'hopcroft karp',
                           'bipartite matching', 'hungarian', 'blossom', 'strongly connected',
                           'articulation', 'bridge', 'tarjan', 'kosaraju', 'centroid',
                           'heavy light', 'binary lifting', 'lowest common ancestor'],
                'weight': 3.8
            },
            'data_structures_advanced': {
                'keywords': ['segment tree', 'fenwick tree', 'binary indexed tree', 'splay tree',
                           'treap', 'suffix array', 'suffix tree', 'trie', 'aho corasick',
                           'persistent', 'wavelet tree', 'skip list', 'rope', 'link-cut tree',
                           'disjoint set union', 'union find', 'dsu on tree'],
                'weight': 3.5
            },
            'mathematics_advanced': {
                'keywords': ['fft', 'ntt', 'modular exponentiation', 'matrix exponentiation',
                           'chinese remainder', 'fermat', 'euler totient', 'miller rabin',
                           'sieve of eratosthenes', 'linear sieve', 'mobius function',
                           'berlekamp massey', 'linear recurrence', 'generating functions',
                           'inclusion exclusion', 'burnside lemma', 'polya enumeration'],
                'weight': 3.7
            },
            'geometry': {
                'keywords': ['convex hull', 'graham scan', 'jarvis march', 'line intersection',
                           'polygon area', 'point in polygon', 'closest pair', 'voronoi',
                           'delaunay triangulation', 'sweep line', 'rotating calipers'],
                'weight': 3.2
            },
            
            # INTERMEDIATE ALGORITHMS (Medium weight)
            'graph_basic': {
                'keywords': ['dijkstra', 'bellman ford', 'floyd warshall', 'topological sort',
                           'bfs', 'dfs', 'minimum spanning', 'kruskal', 'prim', 'euler path',
                           'hamiltonian', 'traveling salesman'],
                'weight': 2.5
            },
            'search_techniques': {
                'keywords': ['binary search', 'ternary search', 'meet in the middle',
                           'two pointer', 'sliding window', 'divide and conquer'],
                'weight': 2.0
            },
            'string_algorithms': {
                'keywords': ['kmp', 'rabin karp', 'z algorithm', 'manacher', 'palindrome',
                           'suffix automation', 'rolling hash', 'trie', 'aho corasick'],
                'weight': 2.3
            },
            
            # BASIC ALGORITHMS (Low weight)
            'basic_ds': {
                'keywords': ['stack', 'queue', 'deque', 'priority queue', 'heap', 'hash',
                           'hashmap', 'hashset', 'linked list', 'array', 'vector'],
                'weight': 1.0
            },
            'basic_math': {
                'keywords': ['gcd', 'lcm', 'prime', 'modulo', 'combinatorics', 'probability',
                           'number theory', 'geometry basics', 'calculus'],
                'weight': 1.5
            }
        }
        
        # Mathematical symbols preserved during preprocessing
        self.math_preserve = {
            'operators': {'+', '-', '*', '/', '=', '^', '%', '**', '//', '±', '∓'},
            'comparisons': {'<', '>', '<=', '>=', '==', '!=', '≡', '≈', '∼', '≠', '≤', '≥', '≪', '≫'},
            'advanced': {'∑', '∏', '∫', '∂', '∇', '∞', '√', '∛', '∜', '→', '↔', '∧', '∨', '⊕', '⊗'},
            'sets': {'∈', '∉', '⊆', '⊂', '∪', '∩', '∅', 'ℕ', 'ℤ', 'ℚ', 'ℝ', 'ℂ'},
            'greek': {'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 
                     'ν', 'ξ', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω'},
        }
        
        # Problem structure indicators
        self.structure_indicators = {
            'sections': ['input', 'output', 'example', 'note', 'constraints', 'explanation',
                        'interaction', 'scoring', 'subtasks', 'time limit', 'memory limit'],
            'list_markers': ['•', '-', '*', '◦', '‣', '·', '▪', '▫'],
            'complexity_terms': ['time complexity', 'space complexity', 'O(', 'Θ(', 'Ω('],
        }
        
        # Compile regex patterns once for efficiency
        self.patterns = {
            'url': re.compile(r'http\S+|www\S+|https\S+'),
            'code_blocks': re.compile(r'```[\s\S]*?```'),
            'inline_code': re.compile(r'`[^`]*`'),
            'numbers': re.compile(r'\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b'),
            'constraints': re.compile(r'(\d+(?:\.\d+)?)\s*(?:≤|<|<=|≤=|less than|up to|maximum|max\.?)\s*[A-Za-z]', re.IGNORECASE),
            'large_numbers': re.compile(r'\b(10\^?\d+|\d+e\d+|\d{4,})\b'),
            'variables': re.compile(r'\b[a-z]\b'),
            'time_limit': re.compile(r'(\d+(?:\.\d+)?)\s*(?:second|sec|s|millisecond|ms)', re.IGNORECASE),
            'memory_limit': re.compile(r'(\d+(?:\.\d+)?)\s*(?:MB|megabyte|GB|gigabyte|KB|kilobyte)', re.IGNORECASE),
        }
        
        # Feature names (FIXED for determinism - 52 features)
        self.feature_names = [
            # Text statistics (6)
            'word_count', 'sentence_count', 'avg_word_length', 
            'unique_word_ratio', 'avg_sentence_length', 'paragraph_count',
            
            # Algorithm features (12)
            'algo_dp_score', 'algo_graph_advanced_score', 'algo_ds_advanced_score',
            'algo_math_advanced_score', 'algo_geometry_score', 'algo_graph_basic_score',
            'algo_search_score', 'algo_string_score', 'algo_basic_ds_score',
            'algo_basic_math_score', 'algo_total_weighted', 'unique_algorithms_count',
            
            # Constraint features (8)
            'max_constraint_log10', 'constraint_count', 'has_large_constraints',
            'constraint_variance', 'constraint_types', 'has_time_limit',
            'has_memory_limit', 'max_time_limit',
            
            # Mathematical features (8)
            'math_symbol_count', 'math_advanced_symbols', 'equation_density',
            'inequality_count', 'summation_integral_count', 'greek_symbol_count',
            'modulo_mentions', 'combinatorics_mentions',
            
            # Structural features (8)
            'section_count', 'example_count', 'has_multiple_test_cases',
            'has_formal_constraints', 'has_subtasks', 'io_format_complexity',
            'interactive_problem', 'has_scoring_section',
            
            # Language complexity features (4)
            'technical_term_density', 'acronym_count', 'conditional_mentions',
            'imperative_verbs',
            
            # Composite scores (6)
            'text_complexity', 'algorithmic_complexity', 'mathematical_complexity',
            'structural_complexity', 'language_complexity', 'overall_complexity_score','score_dp_interaction',
            'score_graph_interaction'
        ]
    
    def deterministic_preprocess(self, text: str) -> str:
        """Deterministic text preprocessing identical for training/inference"""
        if not isinstance(text, str):
            return ""
        
        # 1. Preserve mathematical symbols (add spaces around them)
        preserved = text
        for category in self.math_preserve.values():
            for symbol in category:
                preserved = preserved.replace(symbol, f' {symbol} ')
        
        # 2. Convert to lowercase (except preserved symbols)
        processed = preserved.lower()
        
        # 3. Remove URLs (deterministic)
        processed = self.patterns['url'].sub(' ', processed)
        
        # 4. Remove code blocks
        processed = self.patterns['code_blocks'].sub(' ', processed)
        processed = self.patterns['inline_code'].sub(' ', processed)
        
        # 5. Remove special characters (keep alphanumeric and preserved math)
        processed = re.sub(r'[^\w\s\.\,\!\?\:\;\+\-\*/=<>^%\(\)\[\]\{\}∑∏∫∂∇∞√±→↔∧∨∈∉⊆⊂∪∩∅≤≥≠≈∼≡αβγδεζηθικλμνξπρστυφχψω]', ' ', processed)
        
        # 6. Normalize whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def extract_algorithm_features(self, text: str) -> Dict[str, float]:
        """Extract algorithm presence with hierarchical weights"""
        text_lower = text.lower()
        features = {}
        
        total_weight = 0
        category_scores = {}
        unique_algs = set()
        
        for category, info in self.algorithms.items():
            category_score = 0
            for keyword in info['keywords']:
                # Match whole words
                pattern = rf'\b{re.escape(keyword)}\b'
                matches = re.findall(pattern, text_lower)
                if matches:
                    category_score += len(matches) * info['weight']
                    # Add first word of multi-word keywords
                    if ' ' in keyword:
                        unique_algs.add(keyword.split()[0])
                    else:
                        unique_algs.add(keyword)
            
            category_scores[category] = category_score
            total_weight += category_score
        
        # Store all category scores
        for category in self.algorithms.keys():
            features[f'algo_{category}_score'] = category_scores.get(category, 0)
        
        features['algo_total_weighted'] = total_weight
        features['unique_algorithms_count'] = len(unique_algs)
        
        return features
    
    def extract_constraint_features(self, text: str) -> Dict[str, float]:
        """Extract numerical constraints with magnitude analysis"""
        features = {}
        
        # Find all numbers
        numbers = []
        for match in self.patterns['numbers'].finditer(text):
            try:
                num = float(match.group())
                numbers.append(num)
            except:
                continue
        
        # Find constraints in typical format
        constraint_matches = self.patterns['constraints'].findall(text)
        constraint_values = []
        for match in constraint_matches:
            try:
                val = float(match)
                constraint_values.append(val)
            except:
                continue
        
        all_constraints = numbers + constraint_values
        
        if all_constraints:
            max_constraint = max(all_constraints)
            features['max_constraint_log10'] = np.log10(max_constraint + 1)
            features['constraint_count'] = len(all_constraints)
            features['has_large_constraints'] = 1.0 if max_constraint > 10000 else 0.0
            
            # Constraint variance
            if len(all_constraints) > 1:
                features['constraint_variance'] = np.log1p(np.var(all_constraints))
            else:
                features['constraint_variance'] = 0.0
            
            # Count types of constraints (different orders of magnitude)
            features['constraint_types'] = len(set(int(np.log10(c + 1)) for c in all_constraints if c > 0))
        else:
            features.update({
                'max_constraint_log10': 0,
                'constraint_count': 0,
                'has_large_constraints': 0.0,
                'constraint_variance': 0.0,
                'constraint_types': 0
            })
        
        # Time and memory limits
        time_matches = self.patterns['time_limit'].findall(text)
        memory_matches = self.patterns['memory_limit'].findall(text)
        
        features['has_time_limit'] = 1.0 if time_matches else 0.0
        features['has_memory_limit'] = 1.0 if memory_matches else 0.0
        
        if time_matches:
            try:
                features['max_time_limit'] = max(float(t) for t in time_matches)
            except:
                features['max_time_limit'] = 0.0
        else:
            features['max_time_limit'] = 0.0
        
        return features
    
    def extract_mathematical_features(self, text: str) -> Dict[str, float]:
        """Extract mathematical complexity indicators"""
        features = {}
        
        # Count math symbols by category
        math_count = 0
        advanced_count = 0
        greek_count = 0
        
        for category_name, symbols in self.math_preserve.items():
            for symbol in symbols:
                count = text.count(symbol)
                math_count += count
                if category_name in ['advanced', 'greek']:
                    advanced_count += count
                if category_name == 'greek':
                    greek_count += count
        
        # Count specific mathematical patterns
        equation_patterns = [
            r'[=≠]',  # Equality/inequality
            r'∑', r'∏', r'∫',  # Summations/integrals
            r'[≤≥<>]',  # Inequalities
        ]
        
        equation_density = 0
        inequality_count = 0
        summation_count = 0
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text)
            equation_density += len(matches)
            if pattern in '[≤≥<>]':
                inequality_count += len(matches)
            if pattern in '∑∏∫':
                summation_count += len(matches)
        
        # Count modulo mentions
        modulo_mentions = len(re.findall(r'\bmod\b|\%|modulo', text.lower()))
        
        # Count combinatorics mentions
        combinatorics_mentions = len(re.findall(r'\bpermutation\b|\bcombination\b|\bchoose\b|\bbinomial\b', text.lower()))
        
        features['math_symbol_count'] = math_count
        features['math_advanced_symbols'] = advanced_count
        features['equation_density'] = equation_density / max(1, len(text.split()))
        features['inequality_count'] = inequality_count
        features['summation_integral_count'] = summation_count
        features['greek_symbol_count'] = greek_count
        features['modulo_mentions'] = modulo_mentions
        features['combinatorics_mentions'] = combinatorics_mentions
        
        return features
    
    def extract_structural_features(self, text: str) -> Dict[str, float]:
        """Extract problem structure complexity"""
        features = {}
        lines = text.split('\n')
        
        # Count sections
        section_count = 0
        example_count = 0
        has_constraints = 0
        has_subtasks = 0
        interactive_problem = 0
        has_scoring = 0
        
        for line in lines:
            line_lower = line.lower().strip()
            for section in self.structure_indicators['sections']:
                if line_lower.startswith(section):
                    section_count += 1
                    if section == 'example':
                        example_count += 1
                    elif section == 'constraints':
                        has_constraints = 1
                    elif section == 'subtasks':
                        has_subtasks = 1
                    elif section == 'scoring':
                        has_scoring = 1
                    elif section == 'interaction':
                        interactive_problem = 1
        
        # Check for multiple test cases
        test_case_indicators = ['multiple test cases', 't test cases', 'first line contains t', 'number of test cases']
        has_multiple_tests = 0
        for indicator in test_case_indicators:
            if indicator in text.lower():
                has_multiple_tests = 1
                break
        
        # IO format complexity (count of lines in example input/output)
        io_complexity = 0
        in_example = False
        example_lines = 0
        
        for line in lines:
            if 'example input' in line.lower():
                in_example = True
                example_lines = 0
            elif in_example and ('output' in line.lower() or line.strip() == ''):
                if example_lines > io_complexity:
                    io_complexity = example_lines
                in_example = False
            elif in_example:
                example_lines += 1
        
        # Count paragraphs
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        features['section_count'] = section_count
        features['example_count'] = example_count
        features['has_multiple_test_cases'] = has_multiple_tests
        features['has_formal_constraints'] = has_constraints
        features['has_subtasks'] = has_subtasks
        features['io_format_complexity'] = min(io_complexity, 20) / 20.0  # Normalize
        features['interactive_problem'] = interactive_problem
        features['has_scoring_section'] = has_scoring
        features['paragraph_count'] = paragraph_count
        
        return features
    
    def extract_language_complexity_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic complexity features"""
        words = text.lower().split()
        if not words:
            return {
                'technical_term_density': 0,
                'acronym_count': 0,
                'conditional_mentions': 0,
                'imperative_verbs': 0
            }
        
        # Technical terms (algorithmic/mathematical terms)
        technical_terms = {
            'algorithm', 'complexity', 'optimization', 'efficient', 'optimal',
            'constraint', 'parameter', 'variable', 'function', 'recursive',
            'iterative', 'heuristic', 'deterministic', 'stochastic'
        }
        
        technical_count = sum(1 for word in words if word in technical_terms)
        
        # Acronyms (words in ALL CAPS or with dots)
        acronym_pattern = r'\b[A-Z]{2,}\b|\b[A-Z]\.[A-Z]\.'
        acronym_count = len(re.findall(acronym_pattern, text))
        
        # Conditional statements
        conditional_terms = {'if', 'else', 'when', 'unless', 'provided', 'given that'}
        conditional_count = sum(1 for word in words if word in conditional_terms)
        
        # Imperative verbs (commands)
        imperative_verbs = {'find', 'determine', 'calculate', 'compute', 'print',
                          'output', 'return', 'write', 'implement', 'solve'}
        imperative_count = sum(1 for word in words if word in imperative_verbs)
        
        return {
            'technical_term_density': technical_count / len(words),
            'acronym_count': acronym_count,
            'conditional_mentions': conditional_count,
            'imperative_verbs': imperative_count
        }
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract text complexity features"""
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not words:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'unique_word_ratio': 0,
                'avg_sentence_length': 0
            }
        
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        unique_words = len(set(words))
        
        # Vocabulary richness measures
        ttr = unique_words / word_count  # Type-token ratio
        h_point = 0  # Hapax legomena (words appearing only once)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        h_point = sum(1 for freq in word_freq.values() if freq == 1) / word_count
        
        features = {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': sum(len(w) for w in words) / word_count,
            'unique_word_ratio': ttr * (1 + h_point),  # Enhanced TTR
            'avg_sentence_length': word_count / sentence_count
        }
        
        return features
    
    def extract_composite_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """Compute composite complexity scores"""
        # Text complexity
        text_complexity = (
            0.25 * np.log1p(features.get('word_count', 0)) +
            0.25 * features.get('unique_word_ratio', 0) +
            0.25 * min(features.get('avg_sentence_length', 0) / 25.0, 1.0) +
            0.25 * features.get('technical_term_density', 0)
        )
        
        # Algorithmic complexity (weighted by importance)
        algo_complexity = (
            0.25 * np.log1p(features.get('algo_dp_score', 0)) +
            0.20 * np.log1p(features.get('algo_graph_advanced_score', 0)) +
            0.15 * np.log1p(features.get('algo_math_advanced_score', 0)) +
            0.15 * np.log1p(features.get('algo_ds_advanced_score', 0)) +
            0.10 * np.log1p(features.get('algo_geometry_score', 0)) +
            0.10 * features.get('unique_algorithms_count', 0) / 10.0 +
            0.05 * np.log1p(features.get('algo_total_weighted', 0))
        )
        
        # Mathematical complexity
        math_complexity = (
            0.25 * np.log1p(features.get('math_symbol_count', 0)) +
            0.20 * np.log1p(features.get('math_advanced_symbols', 0)) +
            0.15 * features.get('equation_density', 0) * 10 +
            0.15 * np.log1p(features.get('inequality_count', 0)) +
            0.10 * np.log1p(features.get('summation_integral_count', 0)) +
            0.10 * features.get('modulo_mentions', 0) +
            0.05 * features.get('combinatorics_mentions', 0)
        ) / 2.0
        
        # Structural complexity
        structural_complexity = (
            0.20 * min(features.get('section_count', 0) / 8.0, 1.0) +
            0.20 * min(features.get('example_count', 0) / 3.0, 1.0) +
            0.20 * features.get('io_format_complexity', 0) +
            0.15 * min(features.get('constraint_count', 0) / 10.0, 1.0) +
            0.10 * features.get('has_subtasks', 0) +
            0.10 * features.get('has_multiple_test_cases', 0) +
            0.05 * features.get('interactive_problem', 0)
        )
        
        # Language complexity
        language_complexity = (
            0.40 * features.get('technical_term_density', 0) +
            0.25 * min(features.get('acronym_count', 0) / 5.0, 1.0) +
            0.20 * min(features.get('conditional_mentions', 0) / 5.0, 1.0) +
            0.15 * min(features.get('imperative_verbs', 0) / 10.0, 1.0)
        )
        
        # Overall weighted score (sum to approximately 10 for score prediction)
        overall = (
            0.20 * text_complexity * 2.5 +
            0.35 * algo_complexity * 3.0 +
            0.25 * math_complexity * 2.5 +
            0.12 * structural_complexity * 2.0 +
            0.08 * language_complexity * 1.5
        )
        # --- NEW INTERACTION FEATURES ---
        features['score_dp_interaction'] = (
            features.get('algo_dp_score', 0) * features.get('max_constraint_log10', 0)
        )

        features['score_graph_interaction'] = (
            features.get('algo_graph_advanced_score', 0) * features.get('constraint_count', 0)
        )

        return {
            'text_complexity': text_complexity,
            'algorithmic_complexity': algo_complexity,
            'mathematical_complexity': math_complexity,
            'structural_complexity': structural_complexity,
            'language_complexity': language_complexity,
            'overall_complexity_score': overall
        }
    
    def extract_all_features(self, text: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """Extract all 52 features deterministically"""
        processed_text = self.deterministic_preprocess(text)
        
        # Extract feature categories
        text_features = self.extract_text_features(processed_text)
        algo_features = self.extract_algorithm_features(processed_text)
        constraint_features = self.extract_constraint_features(text)
        math_features = self.extract_mathematical_features(text)
        structural_features = self.extract_structural_features(text)
        language_features = self.extract_language_complexity_features(processed_text)
        
        # Combine all features
        all_features = {}
        all_features.update(text_features)
        all_features.update(algo_features)
        all_features.update(constraint_features)
        all_features.update(math_features)
        all_features.update(structural_features)
        all_features.update(language_features)
        
        # Add composite scores
        composite_scores = self.extract_composite_scores(all_features)
        all_features.update(composite_scores)
        
        # Ensure all features are present and in correct order
        feature_vector = []
        for name in self.feature_names:
            feature_vector.append(all_features.get(name, 0.0))
        
        return np.array(feature_vector).reshape(1, -1), all_features

# ==============================================
# ENHANCED DATA QUALITY ENFORCER
# ==============================================

class EnhancedDataQualityEnforcer:
    """Improved data quality with soft boundaries and better handling"""
    
    def __init__(self):
        self.corrections = []
        self.removals = []
        self.stats = {}
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset statistics"""
        stats = {
            'total_samples': len(df),
            'class_distribution': df['problem_class'].value_counts().to_dict(),
            'score_stats': {
                'mean': df['problem_score'].mean(),
                'std': df['problem_score'].std(),
                'min': df['problem_score'].min(),
                'max': df['problem_score'].max()
            },
            'text_length_stats': {}
        }
        
        # Analyze text lengths
        text_lengths = []
        for idx, row in df.iterrows():
            combined = f"{row['title']} {row['description']} {row['input_description']} {row['output_description']}"
            text_lengths.append(len(combined.split()))
        
        stats['text_length_stats'] = {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': min(text_lengths),
            'max': max(text_lengths)
        }
        
        return stats
    
    def enforce_soft_score_class_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use soft boundaries with probabilistic correction"""
        df_clean = df.copy()
        
        # Normalize class names
        df_clean['problem_class'] = df_clean['problem_class'].apply(
            lambda x: str(x).strip().title() if pd.notnull(x) else 'Medium'
        )
        
        # Define fuzzy boundaries (centroid-based)
        class_centroids = {
            'Easy': 2.5,    # Center around 2.5
            'Medium': 5.5,  # Center around 5.5
            'Hard': 8.5     # Center around 8.5
        }
        
        # Calculate class probabilities based on distance to centroids
        for idx, row in df_clean.iterrows():
            score = row['problem_score']
            actual_class = row['problem_class']
            
            # Calculate distances to each centroid
            distances = {}
            for cls, centroid in class_centroids.items():
                distances[cls] = abs(score - centroid)
            
            # Find closest centroid
            closest_class = min(distances, key=distances.get)
            
            # Only correct if significantly wrong (distance > 2.0)
            if actual_class != closest_class and distances[closest_class] <= 0.8:
                df_clean.at[idx, 'problem_class'] = closest_class
                self.corrections.append({
                    'index': idx,
                    'old_class': actual_class,
                    'new_class': closest_class,
                    'score': score,
                    'action': 'soft_correction',
                    'distance_to_new': distances[closest_class],
                    'distance_to_old': distances[actual_class]
                })
        
        return df_clean
    
    def remove_noisy_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove samples with poor quality text"""
        df_clean = df.copy()
        
        to_remove = []
        for idx, row in df_clean.iterrows():
            combined = f"{row['title']} {row['description']} {row['input_description']} {row['output_description']}"
            words = combined.split()
            
            # Rule 1: Too short
            if len(words) < 15:
                to_remove.append(idx)
                self.removals.append({
                    'index': idx,
                    'reason': 'text_too_short',
                    'word_count': len(words)
                })
                continue
            
            # Rule 2: Too long (potential noise)
            if len(words) > 2000:
                to_remove.append(idx)
                self.removals.append({
                    'index': idx,
                    'reason': 'text_too_long',
                    'word_count': len(words)
                })
                continue
            
            # Rule 3: High repetition
            if len(words) > 50:
                trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                unique_trigrams = len(set(trigrams))
                if unique_trigrams / len(trigrams) < 0.25:
                    to_remove.append(idx)
                    self.removals.append({
                        'index': idx,
                        'reason': 'high_repetition',
                        'unique_trigram_ratio': unique_trigrams / len(trigrams)
                    })
                    continue
            
            # Rule 4: Missing critical sections
            combined_lower = combined.lower()
            has_input = 'input' in combined_lower
            has_output = 'output' in combined_lower
            if not (has_input and has_output):
                to_remove.append(idx)
                self.removals.append({
                    'index': idx,
                    'reason': 'missing_critical_sections',
                    'has_input': has_input,
                    'has_output': has_output
                })
                continue
        
        df_clean = df_clean.drop(to_remove).reset_index(drop=True)
        return df_clean
    
    def aggressive_class_balancing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggressive balancing to handle dataset bias"""
        df_clean = df.copy()
        
        # Normalize class names
        df_clean['problem_class'] = df_clean['problem_class'].apply(
            lambda x: str(x).strip().title() if pd.notnull(x) else 'Medium'
        )
        
        class_counts = df_clean['problem_class'].value_counts()
        
        # Target: equal number of samples per class
        target_count = int(class_counts.median())
        
        balanced_dfs = []
        for cls in df_clean['problem_class'].unique():
            cls_df = df_clean[df_clean['problem_class'] == cls]
            
            if len(cls_df) > target_count:
                # More samples than target - use stratified sampling
                cls_df['score_bin'] = pd.cut(cls_df['problem_score'], bins=5)
                
                sampled = cls_df.groupby('score_bin', group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), max(1, int(target_count / 5))), 
                                      random_state=SEED, replace=False)
                )
                
                # If still not enough, sample more
                if len(sampled) < target_count:
                    additional_needed = target_count - len(sampled)
                    remaining = cls_df.drop(sampled.index)
                    additional = remaining.sample(n=min(additional_needed, len(remaining)), 
                                                 random_state=SEED)
                    sampled = pd.concat([sampled, additional])
                
                cls_df = sampled.drop(columns=['score_bin'])
            elif len(cls_df) < target_count:
                # Fewer samples than target - use SMOTE-like oversampling
                current_len = len(cls_df)
                if current_len > 0:
                    # Simply duplicate with small noise
                    needed = target_count - current_len
                    duplicates = cls_df.sample(n=needed, replace=True, random_state=SEED)
                    # Add small noise to scores
                    cls_df = pd.concat([cls_df, duplicates])
            
            balanced_dfs.append(cls_df)
        
        return pd.concat(balanced_dfs).reset_index(drop=True)
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps with detailed reporting"""
        print("  Step 1: Analyzing dataset...")
        self.stats = self.analyze_dataset(df)
        print(f"    Initial stats: {self.stats}")
        
        print("  Step 2: Enforcing soft score-class consistency...")
        df_clean = self.enforce_soft_score_class_consistency(df)
        
        print("  Step 3: Removing noisy samples...")
        df_clean = self.remove_noisy_samples(df_clean)
        
        print("  Step 4: Aggressive class balancing...")
        df_clean = self.aggressive_class_balancing(df_clean)
        
        # Final stats
        final_stats = self.analyze_dataset(df_clean)
        
        print(f"\n  Cleaning Summary:")
        print(f"    Initial samples: {self.stats['total_samples']}")
        print(f"    Corrections made: {len(self.corrections)}")
        print(f"    Samples removed: {len(self.removals)}")
        print(f"    Final dataset size: {len(df_clean)}")
        print(f"    Class distribution: {final_stats['class_distribution']}")
        print(f"    Score statistics: Mean={final_stats['score_stats']['mean']:.2f}, "
              f"Std={final_stats['score_stats']['std']:.2f}")
        
        return df_clean

# ==============================================
# ENHANCED MODEL ARCHITECTURE
# ==============================================

class EnhancedModelArchitecture:
    """Improved models that address the identified issues"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def create_feature_engineering_pipeline(self, text_processor, df):
        """Create enhanced features"""
        print("  Creating enhanced features...")
        
        # Extract handcrafted features
        numeric_features_list = []
        text_features_list = []
        
        for idx, row in df.iterrows():
            combined_text = (
                f"{row['title']} {row['description']} "
                f"{row['input_description']} {row['output_description']}"
            )
            
            # Extract handcrafted features
            feature_vector, _ = text_processor.extract_all_features(combined_text)
            numeric_features_list.append(feature_vector.flatten())
            
            # Store processed text for embeddings
            processed_text = text_processor.deterministic_preprocess(combined_text)
            text_features_list.append(processed_text)
        
        # TF-IDF with dimensionality reduction
        print("  Creating TF-IDF embeddings...")
        vectorizer = TfidfVectorizer(
            max_features=800,
            min_df=2,
            max_df=0.9,
            stop_words='english',
            sublinear_tf=True,
            ngram_range=(1, 2),
            analyzer='word'
        )
        X_tfidf = vectorizer.fit_transform(text_features_list)
        
        # Dimensionality reduction for TF-IDF
        svd = TruncatedSVD(n_components=200, random_state=self.seed)
        X_tfidf_reduced = svd.fit_transform(X_tfidf)
        
        # Scale numeric features
        scaler = StandardScaler()
        X_numeric = np.array(numeric_features_list)
        X_numeric_scaled = scaler.fit_transform(X_numeric)
        
        # Combine all features
        X_combined = np.hstack([X_tfidf_reduced, X_numeric_scaled])
        
        return X_combined, vectorizer, svd, scaler
    
    def create_hybrid_classifier(self):
        """Create a hybrid classifier (Logistic + XGBoost)"""
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Base models
        logistic = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.seed,
            class_weight='balanced'
        )
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # Voting classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('logistic', logistic),
                ('xgb', xgb_clf)
            ],
            voting='soft',
            weights=[0.25, 0.75]
        )
        
        return voting_clf
    
    def create_enhanced_regressor(self):
        """Create enhanced regressor (Ridge + XGBoost ensemble)"""
        from sklearn.ensemble import StackingRegressor
        
        # Base models
        ridge = Ridge(alpha=1.0, random_state=self.seed)
        
        xgb_reg = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            objective='reg:squarederror'
        )
        
        # Stacking regressor
        stacking_reg = StackingRegressor(
            estimators=[
                ('ridge', ridge),
                ('xgb', xgb_reg)
            ],
            final_estimator=Ridge(alpha=0.5),
            cv=5
        )
        
        return stacking_reg
    
    def train_models(self, X_train, y_class_train, y_score_train):
        """Train both classification and regression models"""
        print("\n  Training hybrid classifier...")
        classifier = self.create_hybrid_classifier()
        classifier.fit(X_train, y_class_train)
        
        print("  Training enhanced regressor...")
        regressor = self.create_enhanced_regressor()
        regressor.fit(X_train, y_score_train)
        
        return classifier, regressor

# ==============================================
# ADVANCED EVALUATION METRICS
# ==============================================

class AdvancedEvaluator:
    """Comprehensive evaluation with bias analysis"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_classification(self, y_true, y_pred, y_proba, class_names):
        """Enhanced classification evaluation"""
        from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Class-wise metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names))
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'per_class_accuracy': dict(zip(class_names, per_class_accuracy)),
            'confusion_matrix': cm,
            'class_report': classification_report(y_true, y_pred, target_names=class_names)
        }
        
        return metrics
    
    def evaluate_regression(self, y_true, y_pred, y_class_true):
        """Enhanced regression evaluation with class analysis"""
        # Overall metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Per-class metrics
        per_class_metrics = {}
        for class_label in np.unique(y_class_true):
            mask = y_class_true == class_label
            if np.sum(mask) > 0:
                per_class_metrics[class_label] = {
                    'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                    'rmse': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
                    'count': np.sum(mask)
                }
        
        # Boundary analysis
        boundaries_crossed = 0
        for y_true_val, y_pred_val in zip(y_true, y_pred):
            true_boundary = self._get_boundary(y_true_val)
            pred_boundary = self._get_boundary(y_pred_val)
            if true_boundary != pred_boundary:
                boundaries_crossed += 1
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'per_class_metrics': per_class_metrics,
            'boundary_consistency': 1 - (boundaries_crossed / len(y_true)),
            'predictions_vs_actual': list(zip(y_true, y_pred))
        }
        
        return metrics
    
    def _get_boundary(self, score):
        """Get class boundary for a score"""
        if score <= 4:
            return 'Easy'
        elif score <= 7:
            return 'Medium'
        else:
            return 'Hard'
    
    def visualize_results(self, y_true, y_pred, y_class_true, y_class_pred, class_names):
        """Create comprehensive visualizations"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Score scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        axes[0, 0].plot([0, 10], [0, 10], 'r--', alpha=0.5, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Score')
        axes[0, 0].set_ylabel('Predicted Score')
        axes[0, 0].set_title('Actual vs Predicted Scores')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Confusion matrix
        cm = confusion_matrix(y_class_true, y_class_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. Score distribution by class
        for i, class_name in enumerate(class_names):
            mask = y_class_true == i
            if np.any(mask):
                axes[1, 0].hist(y_pred[mask], alpha=0.5, label=class_name, bins=20)
        axes[1, 0].set_xlabel('Predicted Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distribution by True Class')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residual plot
        residuals = y_true - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Predicted Score')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../web_app/static/enhanced_results.png', dpi=120, bbox_inches='tight')
        plt.close()
        
        print("Enhanced visualizations saved to web_app/static/enhanced_results.png")

# ==============================================
# MAIN TRAINING PIPELINE
# ==============================================

def main():
    print("="*80)
    print("AUTOJUDGE - ENHANCED TRAINING PIPELINE")
    print("="*80)
    print("\nDesigned to address: Dataset Bias, Model Weakness, and Hard Threshold Issues")
    print(f"Random Seed: {SEED}")
    
    # Step 1: Load Data
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    try:
        data = []
        with open('../data/dataset.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        
        # Rename columns to match expected format
        column_mapping = {
            'input': 'input_description',
            'output': 'output_description',
            'difficulty': 'problem_class',
            'score': 'problem_score'
        }
        df = df.rename(columns=column_mapping)
        
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        print(f"Class distribution: {df['problem_class'].value_counts().to_dict()}")
        
    except FileNotFoundError:
        print("❌ Error: dataset.jsonl not found at ../data/dataset.jsonl")
        print("Please create the dataset first or check the path.")
        return
    
    # Step 2: Enhanced Data Quality Processing
    print("\n" + "="*60)
    print("STEP 2: ENHANCED DATA QUALITY PROCESSING")
    print("="*60)
    
    enforcer = EnhancedDataQualityEnforcer()
    df_clean = enforcer.clean_dataset(df)
    
    # Step 3: Initialize Text Processor
    print("\n" + "="*60)
    print("STEP 3: INITIALIZING TEXT PROCESSOR")
    print("="*60)
    
    processor = UltraEnhancedTextProcessor()
    print(f"Processor initialized with {len(processor.feature_names)} features")
    
    # Step 4: Prepare Data
    print("\n" + "="*60)
    print("STEP 4: DATA PREPARATION")
    print("="*60)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_class = label_encoder.fit_transform(df_clean['problem_class'])
    y_score = df_clean['problem_score'].values
    
    # Create enhanced features
    model_arch = EnhancedModelArchitecture(seed=SEED)
    X, vectorizer, svd, scaler = model_arch.create_feature_engineering_pipeline(processor, df_clean)
    
    # Split data
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2,
        random_state=SEED, stratify=y_class
    )
    
    print(f"\n  Data shapes:")
    print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"    Class distribution in train: {Counter(y_class_train)}")
    print(f"    Class distribution in test: {Counter(y_class_test)}")
    
    # Step 5: Train Enhanced Models
    print("\n" + "="*60)
    print("STEP 5: TRAINING ENHANCED MODELS")
    print("="*60)
    
    classifier, regressor = model_arch.train_models(X_train, y_class_train, y_score_train)
    
    # Step 6: Make Predictions
    print("\n" + "="*60)
    print("STEP 6: MAKING PREDICTIONS")
    print("="*60)
    
    # Get predictions
    y_class_pred = classifier.predict(X_test)
    y_class_proba = classifier.predict_proba(X_test)
    y_score_pred = regressor.predict(X_test)
    # --- CONFIDENCE FILTER ---
    confidence = np.max(y_class_proba, axis=1)

    medium_label = label_encoder.transform(['Medium'])[0]

    for i in range(len(y_class_pred)):
        if confidence[i] < 0.55:
            y_class_pred[i] = medium_label

    # Convert to labels
    y_class_pred_labels = label_encoder.inverse_transform(y_class_pred)
    
    # Step 7: Comprehensive Evaluation
    print("\n" + "="*60)
    print("STEP 7: COMPREHENSIVE EVALUATION")
    print("="*60)
    
    evaluator = AdvancedEvaluator()
    
    # Classification evaluation
    class_metrics = evaluator.evaluate_classification(
        y_class_test, y_class_pred, y_class_proba, label_encoder.classes_
    )
    
    print("\nCLASSIFICATION RESULTS")
    print("="*60)
    print(f"Accuracy: {class_metrics['accuracy']:.4f}")
    print(f"F1 Macro: {class_metrics['f1_macro']:.4f}")
    print(f"F1 Weighted: {class_metrics['f1_weighted']:.4f}")
    print("\nPer-class Accuracy:")
    for cls, acc in class_metrics['per_class_accuracy'].items():
        print(f"  {cls}: {acc:.4f}")
    
    print("\nClassification Report:")
    print(class_metrics['class_report'])
    
    # Regression evaluation
    reg_metrics = evaluator.evaluate_regression(
        y_score_test, y_score_pred, y_class_test
    )
    # --- BOUNDARY ERROR RATE ---
    boundary_errors = 0

    for actual, predicted in zip(y_score_test, y_score_pred):
        if evaluator._get_boundary(actual) != evaluator._get_boundary(predicted):
            boundary_errors += 1

    boundary_error_rate = boundary_errors / len(y_score_test)

    print(f"Boundary Error Rate: {boundary_error_rate:.4f}")

    
    print("\nREGRESSION RESULTS")
    print("="*60)
    print(f"MAE:  {reg_metrics['mae']:.4f}")
    print(f"RMSE: {reg_metrics['rmse']:.4f}")
    print(f"R²:   {reg_metrics['r2']:.4f}")
    print(f"Boundary Consistency: {reg_metrics['boundary_consistency']:.4f}")
    
    print("\nPer-class Regression Metrics:")
    for cls_idx, metrics in reg_metrics['per_class_metrics'].items():
        cls_name = label_encoder.classes_[cls_idx]
        print(f"  {cls_name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, n={metrics['count']}")
    
    # Visualizations
    evaluator.visualize_results(
        y_score_test, y_score_pred, y_class_test, y_class_pred, label_encoder.classes_
    )
    
    # Step 8: Save Models and Artifacts
    print("\n" + "="*60)
    print("STEP 8: SAVING MODELS AND ARTIFACTS")
    print("="*60)
    
    # Create directory
    os.makedirs('saved_models', exist_ok=True)
    
    # Save all artifacts
    artifacts = {
        'classifier': classifier,
        'regressor': regressor,
        'vectorizer': vectorizer,
        'svd': svd,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'text_processor': processor,
        'training_info': {
            'feature_names': processor.feature_names,
            'class_names': list(label_encoder.classes_),
            'num_features': X.shape[1],
            'random_seed': SEED,
            'training_date': pd.Timestamp.now().isoformat(),
            'model_version': '3.0.0',
            'metrics': {
                'classification_accuracy': float(class_metrics['accuracy']),
                'regression_r2': float(reg_metrics['r2']),
                'boundary_consistency': float(reg_metrics['boundary_consistency'])
            }
        }
    }
    
    # Save each artifact
    for name, artifact in artifacts.items():
        if artifact is not None:
            path = f'saved_models/{name}.pkl'
            joblib.dump(artifact, path, compress=3)
            print(f"✅ Saved {name} to {path}")
    
    # Save feature importance if available
    if hasattr(classifier, 'estimators_'):
        # For voting classifier, get feature importance from XGBoost
        for name, est in classifier.named_estimators_.items():
            if hasattr(est, 'feature_importances_'):
                feature_importance = est.feature_importances_
                np.save('saved_models/feature_importance.npy', feature_importance)
                print("✅ Saved feature importance")
                break
    
    print("\n" + "="*80)
    print("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n" + "="*60)
    print("IMPROVEMENTS SUMMARY")
    print("="*60)
    print("✅ Aggressive class balancing to handle Medium-class bias")
    print("✅ Hybrid model architecture (Logistic + XGBoost) for better text understanding")
    print("✅ Soft boundaries instead of hard thresholds")
    print("✅ Enhanced feature engineering with TF-IDF + SVD")
    print("✅ Comprehensive evaluation with boundary consistency analysis")
    print("✅ No manual score adjustments - models learn naturally")
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Classification Accuracy: {class_metrics['accuracy']:.4f}")
    print(f"   Regression R² Score: {reg_metrics['r2']:.4f}")
    print(f"   Boundary Consistency: {reg_metrics['boundary_consistency']:.4f}")
    
    print(f"\n📁 Models saved to: saved_models/")
    print(f"📈 Visualizations saved to: web_app/static/enhanced_results.png")
    
    print("\n🚀 Enhanced system is ready for deployment!")

if __name__ == "__main__":
    main()