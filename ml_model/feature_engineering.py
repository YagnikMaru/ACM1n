import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import List

class FeatureEngineer:
    def __init__(self, max_features=10000):
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def create_tfidf_features(self, texts: List[str], is_train=True):
        """Create TF-IDF features"""
        if is_train:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_features
    
    def create_numeric_features(self, df: pd.DataFrame, is_train=True):
        """Create and scale numeric features"""
        numeric_features = []
        
        for _, row in df.iterrows():
            features = []
            features.append(row['word_count'])
            features.append(row['unique_words'])
            features.append(row['avg_word_length'])
            features.append(row['math_symbol_count'])
            features.append(row['keyword_count'])
            features.append(row['sentence_count'])
            features.append(row['has_recursive'])
            features.append(row['has_dp'])
            features.append(row['has_graph'])
            numeric_features.append(features)
        
        numeric_array = np.array(numeric_features)
        
        if is_train:
            numeric_array = self.scaler.fit_transform(numeric_array)
        else:
            numeric_array = self.scaler.transform(numeric_array)
        
        return numeric_array
    
    def combine_features(self, tfidf_features, numeric_features):
        """Combine TF-IDF and numeric features"""
        from scipy.sparse import hstack
        return hstack([tfidf_features, numeric_features])
    
    def prepare_targets(self, df: pd.DataFrame):
        """Prepare classification and regression targets"""
        # Classification target
        y_class = self.label_encoder.fit_transform(df['problem_class'])
        
        # Regression target (clipped to 0-10)
        y_score = df['problem_score'].values
        y_score = np.clip(y_score, 0, 10)
        
        return y_class, y_score
    
    def save_artifacts(self, path='saved_models/'):
        """Save feature engineering artifacts"""
        joblib.dump(self.tfidf_vectorizer, f'{path}/vectorizer.pkl')
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.label_encoder, f'{path}/label_encoder.pkl')
    
    def load_artifacts(self, path='saved_models/'):
        """Load feature engineering artifacts"""
        self.tfidf_vectorizer = joblib.load(f'{path}/vectorizer.pkl')
        self.scaler = joblib.load(f'{path}/scaler.pkl')
        self.label_encoder = joblib.load(f'{path}/label_encoder.pkl')