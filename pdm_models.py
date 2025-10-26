# pdm_models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PDMPredictiveModels:
    """
    Predictive models for PDM (Predictive Data Maintenance) dataset.
    Includes both traditional ML and deep learning models.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.sequence_length = 30
        
    def build_cnn_lstm_model(self, input_shape, n_features):
        """
        Build CNN-LSTM model for sequence-based failure prediction.
        """
        model = Sequential([
            # Convolutional layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # LSTM layers for temporal dependencies
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            
            # Dense layers for classification
            Dense(50, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(25, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_lstm_model(self, input_shape, n_features):
        """
        Build pure LSTM model for sequence-based failure prediction.
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_random_forest(self, X_train, y_train):
        """
        Build and optimize Random Forest model.
        """
        print("🔄 Building Random Forest model...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        
        print(f"✅ Best Random Forest parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
        
        return best_rf
    
    def build_gradient_boosting(self, X_train, y_train):
        """
        Build and optimize Gradient Boosting model.
        """
        print("🔄 Building Gradient Boosting model...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_gb = grid_search.best_estimator_
        
        print(f"✅ Best Gradient Boosting parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
        
        return best_gb
    
    def build_logistic_regression(self, X_train, y_train):
        """
        Build and optimize Logistic Regression model.
        """
        print("🔄 Building Logistic Regression model...")
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(
            lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_lr = grid_search.best_estimator_
        
        print(f"✅ Best Logistic Regression parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
        
        return best_lr
    
    def train_sequence_models(self, X_seq, y_seq, validation_split=0.2, epochs=100):
        """
        Train sequence-based models (CNN-LSTM and LSTM).
        """
        print("🔄 Training sequence-based models...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=validation_split, random_state=42, stratify=y_seq
        )
        
        # Train CNN-LSTM
        print("\n🤖 Training CNN-LSTM model...")
        cnn_lstm_model = self.build_cnn_lstm_model(
            (X_seq.shape[1], X_seq.shape[2]), X_seq.shape[2]
        )
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001),
            ModelCheckpoint('best_cnn_lstm_model.keras', save_best_only=True)
        ]
        
        cnn_lstm_history = cnn_lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train LSTM
        print("\n🤖 Training LSTM model...")
        lstm_model = self.build_lstm_model(
            (X_seq.shape[1], X_seq.shape[2]), X_seq.shape[2]
        )
        
        callbacks_lstm = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001),
            ModelCheckpoint('best_lstm_model.keras', save_best_only=True)
        ]
        
        lstm_history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks_lstm,
            verbose=1
        )
        
        # Evaluate models
        self.evaluate_sequence_models(cnn_lstm_model, lstm_model, X_val, y_val)
        
        self.models['cnn_lstm'] = cnn_lstm_model
        self.models['lstm'] = lstm_model
        
        return cnn_lstm_history, lstm_history
    
    def train_tabular_models(self, X_tab, y_tab):
        """
        Train tabular models (Random Forest, Gradient Boosting, Logistic Regression).
        """
        print("🔄 Training tabular models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tab, y_tab, test_size=0.2, random_state=42, stratify=y_tab
        )
        
        # Train Random Forest
        rf_model = self.build_random_forest(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # Train Gradient Boosting
        gb_model = self.build_gradient_boosting(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        
        # Train Logistic Regression
        lr_model = self.build_logistic_regression(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        
        # Evaluate models
        self.evaluate_tabular_models(X_test, y_test)
        
        return X_test, y_test
    
    def evaluate_sequence_models(self, cnn_lstm_model, lstm_model, X_val, y_val):
        """
        Evaluate sequence-based models.
        """
        print("\n📊 Evaluating sequence models...")
        
        # CNN-LSTM evaluation
        cnn_lstm_pred = cnn_lstm_model.predict(X_val)
        cnn_lstm_pred_binary = (cnn_lstm_pred > 0.5).astype(int)
        
        print("\n🔍 CNN-LSTM Results:")
        print(classification_report(y_val, cnn_lstm_pred_binary))
        print(f"ROC-AUC: {roc_auc_score(y_val, cnn_lstm_pred):.4f}")
        
        # LSTM evaluation
        lstm_pred = lstm_model.predict(X_val)
        lstm_pred_binary = (lstm_pred > 0.5).astype(int)
        
        print("\n🔍 LSTM Results:")
        print(classification_report(y_val, lstm_pred_binary))
        print(f"ROC-AUC: {roc_auc_score(y_val, lstm_pred):.4f}")
    
    def evaluate_tabular_models(self, X_test, y_test):
        """
        Evaluate tabular models.
        """
        print("\n📊 Evaluating tabular models...")
        
        for name, model in self.models.items():
            if name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                print(f"\n🔍 {name.replace('_', ' ').title()} Results:")
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                print(classification_report(y_test, y_pred))
                print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    self.plot_feature_importance(model, name)
    
    def plot_feature_importance(self, model, model_name):
        """
        Plot feature importance for tree-based models.
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[-20:]  # Top 20 features
            
            plt.figure(figsize=(10, 8))
            plt.title(f'{model_name.replace("_", " ").title()} - Feature Importance')
            plt.barh(range(len(indices)), importance[indices])
            plt.yticks(range(len(indices)), [self.feature_columns[i] for i in indices])
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_training_history(self, cnn_lstm_history, lstm_history):
        """
        Plot training history for sequence models.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CNN-LSTM history
        axes[0, 0].plot(cnn_lstm_history.history['loss'], label='Training Loss')
        axes[0, 0].plot(cnn_lstm_history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('CNN-LSTM Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(cnn_lstm_history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(cnn_lstm_history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('CNN-LSTM Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # LSTM history
        axes[1, 0].plot(lstm_history.history['loss'], label='Training Loss')
        axes[1, 0].plot(lstm_history.history['val_loss'], label='Validation Loss')
        axes[1, 0].set_title('LSTM Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(lstm_history.history['accuracy'], label='Training Accuracy')
        axes[1, 1].plot(lstm_history.history['val_accuracy'], label='Validation Accuracy')
        axes[1, 1].set_title('LSTM Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def ensemble_predict(self, X_tab, X_seq=None):
        """
        Make ensemble predictions using all models.
        """
        predictions = {}
        probabilities = {}
        
        # Tabular model predictions
        for name, model in self.models.items():
            if name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                pred = model.predict(X_tab)
                pred_proba = model.predict_proba(X_tab)[:, 1]
                predictions[name] = pred
                probabilities[name] = pred_proba
        
        # Sequence model predictions
        if X_seq is not None:
            for name, model in self.models.items():
                if name in ['cnn_lstm', 'lstm']:
                    pred_proba = model.predict(X_seq)
                    pred = (pred_proba > 0.5).astype(int)
                    predictions[name] = pred.flatten()
                    probabilities[name] = pred_proba.flatten()
        
        # Ensemble prediction (average of probabilities)
        if probabilities:
            ensemble_proba = np.mean(list(probabilities.values()), axis=0)
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            predictions['ensemble'] = ensemble_pred
            probabilities['ensemble'] = ensemble_proba
        
        return predictions, probabilities
    
    def save_models(self, model_dir='pdm_models/'):
        """
        Save all trained models.
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save tabular models
        for name, model in self.models.items():
            if name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                joblib.dump(model, f'{model_dir}{name}_model.pkl')
                print(f"✅ {name} model saved!")
        
        # Save sequence models
        for name, model in self.models.items():
            if name in ['cnn_lstm', 'lstm']:
                model.save(f'{model_dir}{name}_model.keras')
                print(f"✅ {name} model saved!")
        
        # Save processor info
        processor_info = {
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length
        }
        joblib.dump(processor_info, f'{model_dir}processor_info.pkl')
        print("✅ Processor info saved!")

if __name__ == '__main__':
    # This will be used when training models
    print("PDM Predictive Models class loaded successfully!")
    print("Use this class to train models on PDM data.")
