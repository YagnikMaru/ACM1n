import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer

def train_models():
    print("Starting AutoJudge Model Training...")
    print("="*50)
    
    # Initialize classes
    preprocessor = TextPreprocessor()
    feature_engineer = FeatureEngineer(max_features=5000)
    
    # Load and preprocess data
    print("1. Loading and preprocessing data...")
    df = preprocessor.load_data('../data/dataset.jsonl')
    
    # Combine text fields
    df['combined_text'] = df.apply(preprocessor.combine_text_fields, axis=1)
    df['processed_text'] = df['combined_text'].apply(preprocessor.preprocess_text)
    
    # Extract features
    print("2. Extracting features...")
    text_features = []
    for _, row in df.iterrows():
        features = preprocessor.extract_text_features(row['processed_text'])
        text_features.append(features)
    
    features_df = pd.DataFrame(text_features)
    df = pd.concat([df, features_df], axis=1)
    
    # Prepare features
    print("3. Creating feature matrix...")
    X_tfidf = feature_engineer.create_tfidf_features(df['processed_text'].tolist(), is_train=True)
    X_numeric = feature_engineer.create_numeric_features(df, is_train=True)
    X = feature_engineer.combine_features(X_tfidf, X_numeric)
    
    # Prepare targets
    y_class, y_score = feature_engineer.prepare_targets(df)
    
    # Split data
    print("4. Splitting data...")
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Train Classification Model
    print("\n5. Training Classification Model...")
    print("-"*30)
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    best_classifier = None
    best_accuracy = 0
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_class_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_class_test, y_pred)
        print(f"  Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = clf
    
    print(f"\nBest Classifier: {type(best_classifier).__name__}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    # Classification evaluation
    y_class_pred = best_classifier.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_class_test, y_class_pred, 
                                target_names=feature_engineer.label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_class_test, y_class_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=feature_engineer.label_encoder.classes_,
                yticklabels=feature_engineer.label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('../web_app/static/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Train Regression Model
    print("\n6. Training Regression Model...")
    print("-"*30)
    
    regressors = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_regressor = None
    best_mae = float('inf')
    
    for name, reg in regressors.items():
        print(f"Training {name}...")
        reg.fit(X_train, y_score_train)
        y_pred = reg.predict(X_test)
        mae = mean_absolute_error(y_score_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_score_test, y_pred))
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_regressor = reg
    
    print(f"\nBest Regressor: {type(best_regressor).__name__}")
    print(f"Best MAE: {best_mae:.4f}")
    
    # Regression evaluation
    y_score_pred = best_regressor.predict(X_test)
    y_score_pred = np.clip(y_score_pred, 0, 10)
    
    # Plot predictions vs actual
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
    print("\n7. Saving models and artifacts...")
    joblib.dump(best_classifier, 'saved_models/classifier.pkl')
    joblib.dump(best_regressor, 'saved_models/regressor.pkl')
    feature_engineer.save_artifacts('saved_models/')
    
    print("\n8. Final Evaluation Metrics:")
    print("="*50)
    print(f"Classification Accuracy: {best_accuracy:.4f}")
    print(f"Regression MAE: {best_mae:.4f}")
    print(f"Regression RMSE: {np.sqrt(mean_squared_error(y_score_test, y_score_pred)):.4f}")
    
    # Print sample predictions
    print("\n9. Sample Predictions:")
    print("-"*30)
    sample_indices = np.random.choice(len(y_score_test), min(5, len(y_score_test)), replace=False)
    for idx in sample_indices:
        actual_class = feature_engineer.label_encoder.inverse_transform([y_class_test[idx]])[0]
        pred_class = feature_engineer.label_encoder.inverse_transform([y_class_pred[idx]])[0]
        print(f"Sample {idx}:")
        print(f"  Actual: Class={actual_class}, Score={y_score_test[idx]:.2f}")
        print(f"  Predicted: Class={pred_class}, Score={y_score_pred[idx]:.2f}")
        print()
    
    print("\nTraining completed successfully!")
    print(f"Models saved to: saved_models/")
    print(f"Evaluation plots saved to: web_app/static/")

if __name__ == "__main__":
    train_models()