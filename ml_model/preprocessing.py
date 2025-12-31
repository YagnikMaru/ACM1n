import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict

class TextPreprocessor:
    def __init__(self):
        # Keywords to preserve
        self.keywords = [
            'dp', 'graph', 'tree', 'greedy', 'math', 'recursion', 'bitmask',
            'bfs', 'dfs', 'binary search', 'dynamic programming', 'backtracking',
            'dijkstra', 'sorting', 'stack', 'queue', 'hash', 'map', 'array',
            'string', 'linked list', 'matrix', 'heap', 'priority queue'
        ]
        
        # Mathematical symbols to preserve
        self.math_symbols = {'+', '-', '*', '/', '^', '<=', '>=', '==', '!=', '=', '<', '>'}
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load JSONL data"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return pd.DataFrame(data)
    
    def combine_text_fields(self, row: pd.Series) -> str:
        """Combine all text fields into one"""
        text_parts = [
            str(row.get('title', '')),
            str(row.get('description', '')),
            str(row.get('input_description', '')),
            str(row.get('output_description', ''))
        ]
        return ' '.join([part for part in text_parts if part and str(part).strip()])
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Preserve mathematical symbols by adding spaces
        for symbol in self.math_symbols:
            text = text.replace(symbol, f' {symbol} ')
        
        # Remove special characters except preserved symbols
        text = re.sub(r'[^\w\s\+\-\*/^<>=]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_text_features(self, text: str) -> Dict:
        """Extract basic text features"""
        features = {}
        
        # Word count
        words = text.split()
        features['word_count'] = len(words)
        
        # Unique words
        features['unique_words'] = len(set(words))
        
        # Average word length
        if words:
            features['avg_word_length'] = sum(len(w) for w in words) / len(words)
        else:
            features['avg_word_length'] = 0
        
        # Count mathematical symbols
        math_count = 0
        for symbol in self.math_symbols:
            math_count += text.count(symbol)
        features['math_symbol_count'] = math_count
        
        # Count programming keywords
        keyword_count = 0
        for keyword in self.keywords:
            if keyword in text:
                keyword_count += 1
        features['keyword_count'] = keyword_count
        
        # Sentence count (approximate)
        sentence_enders = ['.', '?', '!']
        features['sentence_count'] = sum(text.count(ender) for ender in sentence_enders)
        
        # Complexity indicators
        features['has_recursive'] = 1 if 'recursion' in text or 'recursive' in text else 0
        features['has_dp'] = 1 if 'dp' in text or 'dynamic' in text else 0
        features['has_graph'] = 1 if 'graph' in text or 'node' in text or 'edge' in text else 0
        
        return features