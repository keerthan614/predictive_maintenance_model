# pdm_inference.py
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PDMInference:
    """
    Real-time inference pipeline for PDM (Predictive Data Maintenance) system.
    Handles both tabular and sequence-based predictions.
    """
    
    def __init__(self, model_dir='pdm_models/'):
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.sequence_length = 30
        self.load_models()
        
    def load_models(self):
        """Load all trained models and processor info."""
        print("🔄 Loading PDM models...")
        
        try:
            # Load processor info
            processor_info = joblib.load(f'{self.model_dir}processor_info.pkl')
            self.feature_columns = processor_info['feature_columns']
            self.sequence_length = processor_info['sequence_length']
            print("✅ Processor info loaded!")
        except Exception as e:
            print(f"❌ Failed to load processor info: {e}")
        
        # Load tabular models
        tabular_models = ['random_forest', 'gradient_boosting', 'logistic_regression']
        for model_name in tabular_models:
            try:
                self.models[model_name] = joblib.load(f'{self.model_dir}{model_name}_model.pkl')
                print(f"✅ {model_name} model loaded!")
            except Exception as e:
                print(f"❌ Failed to load {model_name} model: {e}")
        
        # Load sequence models
        sequence_models = ['cnn_lstm', 'lstm']
        for model_name in sequence_models:
            try:
                self.models[model_name] = tf.keras.models.load_model(f'{self.model_dir}{model_name}_model.keras')
                print(f"✅ {model_name} model loaded!")
            except Exception as e:
                try:
                    self.models[model_name] = tf.keras.models.load_model(f'{self.model_dir}{model_name}_model.h5')
                    print(f"✅ {model_name} model loaded!")
                except Exception as e2:
                    print(f"❌ Failed to load {model_name} model: {e2}")
    
    def preprocess_telemetry_data(self, telemetry_data: Dict) -> np.ndarray:
        """
        Preprocess real-time telemetry data for tabular models.
        
        Args:
            telemetry_data: Dict with sensor readings
            Example: {
                'volt': 175.5,
                'rotate': 420.3,
                'pressure': 110.2,
                'vibration': 45.1,
                'machineID': 1,
                'model': 'model3',
                'age': 18
            }
        
        Returns:
            Preprocessed feature array
        """
        # Convert to DataFrame
        df = pd.DataFrame([telemetry_data])
        
        # Create rolling window features (simplified for single data point)
        features_to_roll = ['volt', 'rotate', 'pressure', 'vibration']
        
        for feature in features_to_roll:
            if feature in df.columns:
                df[f'{feature}_rolling_mean_3h'] = df[feature]
                df[f'{feature}_rolling_std_3h'] = 0
                df[f'{feature}_rolling_mean_24h'] = df[feature]
                df[f'{feature}_rolling_std_24h'] = 0
                df[f'{feature}_lag_1h'] = df[feature]
                df[f'{feature}_lag_3h'] = df[feature]
                df[f'{feature}_lag_24h'] = df[feature]
                df[f'{feature}_diff_1h'] = 0
                df[f'{feature}_diff_3h'] = 0
        
        # Encode categorical features
        if 'model' in df.columns:
            # Simple encoding for demo (in production, use saved encoders)
            model_mapping = {'model1': 0, 'model2': 1, 'model3': 2, 'model4': 3}
            df['model_encoded'] = df['model'].map(model_mapping).fillna(0)
        
        # Add error and maintenance features (set to 0 for single prediction)
        error_cols = [col for col in self.feature_columns if col.startswith('error')]
        maint_cols = [col for col in self.feature_columns if col.startswith('comp')]
        
        for col in error_cols + maint_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data
        df = df[self.feature_columns]
        
        return df.values
    
    def create_sequence_data(self, historical_data: List[Dict]) -> np.ndarray:
        """
        Create sequence data for sequence-based models.
        
        Args:
            historical_data: List of telemetry data dictionaries (most recent last)
        
        Returns:
            Sequence array for sequence models
        """
        if len(historical_data) < self.sequence_length:
            # Pad with the most recent data if not enough history
            padded_data = historical_data[-1:] * (self.sequence_length - len(historical_data)) + historical_data
        else:
            padded_data = historical_data[-self.sequence_length:]
        
        # Preprocess each data point
        processed_data = []
        for data_point in padded_data:
            processed = self.preprocess_telemetry_data(data_point)
            processed_data.append(processed[0])  # Remove batch dimension
        
        return np.array(processed_data).reshape(1, self.sequence_length, -1)
    
    def predict_failure_tabular(self, telemetry_data: Dict) -> Dict:
        """
        Predict failure using tabular models.
        
        Args:
            telemetry_data: Current telemetry readings
        
        Returns:
            Dict with prediction results
        """
        try:
            # Preprocess data
            X_processed = self.preprocess_telemetry_data(telemetry_data)
            
            # Get predictions from all tabular models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                if name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                    pred = model.predict(X_processed)[0]
                    pred_proba = model.predict_proba(X_processed)[0][1]
                    predictions[name] = pred
                    probabilities[name] = pred_proba
            
            # Ensemble prediction
            if probabilities:
                ensemble_proba = np.mean(list(probabilities.values()))
                ensemble_pred = int(ensemble_proba > 0.5)
                predictions['ensemble'] = ensemble_pred
                probabilities['ensemble'] = ensemble_proba
            
            # Determine confidence and urgency
            max_proba = max(probabilities.values()) if probabilities else 0
            confidence = "high" if max_proba > 0.8 else "medium" if max_proba > 0.5 else "low"
            urgency = "critical" if max_proba > 0.8 else "warning" if max_proba > 0.5 else "normal"
            
            # Generate recommendations
            recommendations = []
            if max_proba > 0.8:
                recommendations.extend([
                    "🚨 IMMEDIATE ACTION REQUIRED: Schedule maintenance immediately",
                    "📞 Contact maintenance team and prepare spare parts",
                    "🔍 Check for specific component failures"
                ])
            elif max_proba > 0.5:
                recommendations.extend([
                    "⚠️ Schedule maintenance within 1-2 weeks",
                    "📋 Prepare maintenance checklist and order spare parts",
                    "📊 Monitor sensor readings more frequently"
                ])
            else:
                recommendations.extend([
                    "✅ System operating normally",
                    "📊 Continue regular monitoring",
                    "📅 Schedule routine maintenance as planned"
                ])
            
            return {
                "failure_probability": float(max_proba),
                "predictions": predictions,
                "probabilities": probabilities,
                "confidence": confidence,
                "urgency_level": urgency,
                "recommendations": recommendations,
                "model_type": "tabular"
            }
            
        except Exception as e:
            return {"error": f"Tabular prediction failed: {str(e)}"}
    
    def predict_failure_sequence(self, historical_data: List[Dict]) -> Dict:
        """
        Predict failure using sequence-based models.
        
        Args:
            historical_data: Historical telemetry readings
        
        Returns:
            Dict with prediction results
        """
        try:
            # Create sequence data
            X_seq = self.create_sequence_data(historical_data)
            
            # Get predictions from sequence models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                if name in ['cnn_lstm', 'lstm']:
                    pred_proba = model.predict(X_seq, verbose=0)[0][0]
                    pred = int(pred_proba > 0.5)
                    predictions[name] = pred
                    probabilities[name] = float(pred_proba)
            
            # Ensemble prediction
            if probabilities:
                ensemble_proba = np.mean(list(probabilities.values()))
                ensemble_pred = int(ensemble_proba > 0.5)
                predictions['ensemble'] = ensemble_pred
                probabilities['ensemble'] = ensemble_proba
            
            # Determine confidence and urgency
            max_proba = max(probabilities.values()) if probabilities else 0
            confidence = "high" if max_proba > 0.8 else "medium" if max_proba > 0.5 else "low"
            urgency = "critical" if max_proba > 0.8 else "warning" if max_proba > 0.5 else "normal"
            
            return {
                "failure_probability": float(max_proba),
                "predictions": predictions,
                "probabilities": probabilities,
                "confidence": confidence,
                "urgency_level": urgency,
                "model_type": "sequence"
            }
            
        except Exception as e:
            return {"error": f"Sequence prediction failed: {str(e)}"}
    
    def predict_failure_ensemble(self, telemetry_data: Dict, historical_data: List[Dict] = None) -> Dict:
        """
        Make ensemble prediction using both tabular and sequence models.
        
        Args:
            telemetry_data: Current telemetry readings
            historical_data: Historical telemetry readings (optional)
        
        Returns:
            Dict with ensemble prediction results
        """
        # Get tabular prediction
        tabular_result = self.predict_failure_tabular(telemetry_data)
        
        # Get sequence prediction if historical data is available
        sequence_result = None
        if historical_data and len(historical_data) > 0:
            sequence_result = self.predict_failure_sequence(historical_data)
        
        # Combine results
        all_probabilities = {}
        all_predictions = {}
        
        if 'probabilities' in tabular_result:
            all_probabilities.update(tabular_result['probabilities'])
            all_predictions.update(tabular_result['predictions'])
        
        if sequence_result and 'probabilities' in sequence_result:
            all_probabilities.update(sequence_result['probabilities'])
            all_predictions.update(sequence_result['predictions'])
        
        if not all_probabilities:
            return {"error": "No valid predictions available"}
        
        # Ensemble prediction
        ensemble_proba = np.mean(list(all_probabilities.values()))
        ensemble_pred = int(ensemble_proba > 0.5)
        
        # Determine confidence based on agreement between models
        if len(all_probabilities) > 1:
            prob_values = list(all_probabilities.values())
            agreement = 1 - (np.std(prob_values) / (np.mean(prob_values) + 1e-8))
            confidence = "high" if agreement > 0.8 else "medium" if agreement > 0.6 else "low"
        else:
            confidence = "medium"
        
        urgency = "critical" if ensemble_proba > 0.8 else "warning" if ensemble_proba > 0.5 else "normal"
        
        # Generate recommendations
        recommendations = []
        if ensemble_proba > 0.8:
            recommendations.extend([
                "🚨 IMMEDIATE ACTION REQUIRED: Schedule maintenance immediately",
                "📞 Contact maintenance team and prepare spare parts",
                "🔍 Check for specific component failures",
                "📊 Both tabular and sequence models agree on high failure risk"
            ])
        elif ensemble_proba > 0.5:
            recommendations.extend([
                "⚠️ Schedule maintenance within 1-2 weeks",
                "📋 Prepare maintenance checklist and order spare parts",
                "📊 Monitor sensor readings more frequently",
                "🔄 Consider running additional diagnostics"
            ])
        else:
            recommendations.extend([
                "✅ System operating normally",
                "📊 Continue regular monitoring",
                "📅 Schedule routine maintenance as planned"
            ])
        
        return {
            "failure_probability": float(ensemble_proba),
            "ensemble_prediction": ensemble_pred,
            "individual_predictions": all_predictions,
            "individual_probabilities": all_probabilities,
            "confidence": confidence,
            "urgency_level": urgency,
            "recommendations": recommendations,
            "model_types_used": ["tabular"] + (["sequence"] if sequence_result else []),
            "model_type": "ensemble"
        }
    
    def batch_predict(self, telemetry_batch: List[Dict], historical_batch: List[List[Dict]] = None) -> List[Dict]:
        """
        Make predictions for a batch of telemetry readings.
        
        Args:
            telemetry_batch: List of telemetry data dictionaries
            historical_batch: List of historical data for each machine (optional)
        
        Returns:
            List of prediction results
        """
        results = []
        
        for i, telemetry_data in enumerate(telemetry_batch):
            historical_data = historical_batch[i] if historical_batch and i < len(historical_batch) else None
            
            result = self.predict_failure_ensemble(telemetry_data, historical_data)
            results.append({
                "machine_id": telemetry_data.get('machineID', f'machine_{i+1}'),
                "telemetry_data": telemetry_data,
                "prediction": result
            })
        
        return results

def create_sample_telemetry_data(machine_id: int = 1) -> Dict:
    """Create sample telemetry data for testing."""
    return {
        'volt': np.random.normal(175, 10),
        'rotate': np.random.normal(420, 20),
        'pressure': np.random.normal(110, 5),
        'vibration': np.random.normal(45, 3),
        'machineID': machine_id,
        'model': np.random.choice(['model1', 'model2', 'model3', 'model4']),
        'age': np.random.randint(1, 20)
    }

def create_sample_historical_data(machine_id: int = 1, n_points: int = 35) -> List[Dict]:
    """Create sample historical telemetry data for testing."""
    historical_data = []
    base_time = pd.Timestamp.now() - pd.Timedelta(hours=n_points)
    
    for i in range(n_points):
        # Simulate gradual degradation
        data = {
            'volt': np.random.normal(175 - i*0.5, 10),
            'rotate': np.random.normal(420 - i*2, 20),
            'pressure': np.random.normal(110 + i*0.3, 5),
            'vibration': np.random.normal(45 + i*0.8, 3),
            'machineID': machine_id,
            'model': 'model3',
            'age': 18,
            'datetime': base_time + pd.Timedelta(hours=i)
        }
        historical_data.append(data)
    
    return historical_data

if __name__ == '__main__':
    # Test the inference pipeline
    print("🔄 Testing PDM Inference Pipeline...")
    
    # Initialize inference
    inference = PDMInference()
    
    # Test with sample data
    print("\n🧪 Testing with sample telemetry data...")
    
    # Single prediction
    telemetry_data = create_sample_telemetry_data()
    print(f"📊 Current telemetry data: {telemetry_data}")
    
    result = inference.predict_failure_tabular(telemetry_data)
    print(f"\n🎯 Tabular Prediction Result:")
    if 'error' not in result:
        print(f"   Failure Probability: {result.get('failure_probability', 'N/A'):.3f}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        print(f"   Urgency: {result.get('urgency_level', 'N/A')}")
        print(f"\n💡 Recommendations:")
        for rec in result.get('recommendations', []):
            print(f"   {rec}")
    else:
        print(f"   Error: {result['error']}")
    
    # Ensemble prediction with historical data
    print("\n🔄 Testing Ensemble Prediction with Historical Data...")
    historical_data = create_sample_historical_data()
    ensemble_result = inference.predict_failure_ensemble(telemetry_data, historical_data)
    
    print(f"🎯 Ensemble Prediction Result:")
    if 'error' not in ensemble_result:
        print(f"   Failure Probability: {ensemble_result.get('failure_probability', 'N/A'):.3f}")
        print(f"   Confidence: {ensemble_result.get('confidence', 'N/A')}")
        print(f"   Urgency: {ensemble_result.get('urgency_level', 'N/A')}")
        print(f"   Models Used: {ensemble_result.get('model_types_used', 'N/A')}")
    else:
        print(f"   Error: {ensemble_result['error']}")
    
    # Batch prediction test
    print("\n🔄 Testing Batch Prediction...")
    batch_telemetry = [create_sample_telemetry_data(i) for i in range(1, 4)]
    batch_historical = [create_sample_historical_data(i) for i in range(1, 4)]
    batch_results = inference.batch_predict(batch_telemetry, batch_historical)
    
    print(f"\n📊 Batch Prediction Results:")
    for result in batch_results:
        pred = result['prediction']
        if 'error' not in pred:
            print(f"   {result['machine_id']}: Prob={pred.get('failure_probability', 'N/A'):.3f}, "
                  f"Urgency={pred.get('urgency_level', 'N/A')}")
        else:
            print(f"   {result['machine_id']}: Error - {pred['error']}")
    
    print("\n✅ PDM Inference Pipeline test completed!")
