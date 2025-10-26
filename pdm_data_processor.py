# pdm_data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PDMDataProcessor:
    """
    Data processor for Predictive Data Maintenance (PDM) dataset.
    Handles telemetry, machines, errors, failures, and maintenance data.
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.sequence_length = 30
        
    def load_data(self, data_dir='./'):
        """
        Load all PDM datasets.
        """
        print("🔄 Loading PDM datasets...")
        data = {}
        data['telemetry'] = pd.read_csv(f'{data_dir}PdM_telemetry.csv', nrows=10000)
        data['telemetry']['datetime'] = pd.to_datetime(data['telemetry']['datetime'])
        print(f"✅ Telemetry data: {data['telemetry'].shape}")
        
        data['machines'] = pd.read_csv(f'{data_dir}PdM_machines.csv')
        print(f"✅ Machines data: {data['machines'].shape}")
        
        data['errors'] = pd.read_csv(f'{data_dir}PdM_errors.csv')
        data['errors']['datetime'] = pd.to_datetime(data['errors']['datetime'])
        print(f"✅ Errors data: {data['errors'].shape}")
        
        data['failures'] = pd.read_csv(f'{data_dir}PdM_failures.csv')
        data['failures']['datetime'] = pd.to_datetime(data['failures']['datetime'])
        print(f"✅ Failures data: {data['failures'].shape}")
        
        data['maint'] = pd.read_csv(f'{data_dir}PdM_maint.csv')
        data['maint']['datetime'] = pd.to_datetime(data['maint']['datetime'])
        print(f"✅ Maintenance data: {data['maint'].shape}")
        
        return data
    
    def create_telemetry_features(self, telemetry_df):
        """
        Create time-series features from telemetry data.
        """
        print("🔄 Creating telemetry features...")
        telemetry_df = telemetry_df.sort_values(['machineID', 'datetime'])
        
        for col in ['volt', 'rotate', 'pressure', 'vibration']:
            telemetry_df[f'{col}_rolling_mean_24h'] = telemetry_df.groupby('machineID')[col].rolling(window=24, min_periods=1).mean().reset_index(level=0, drop=True)
            telemetry_df[f'{col}_rolling_std_24h'] = telemetry_df.groupby('machineID')[col].rolling(window=24, min_periods=1).std().reset_index(level=0, drop=True)
        
        telemetry_df = telemetry_df.fillna(method='bfill').fillna(method='ffill')
        print(f"✅ Telemetry features created: {telemetry_df.shape}")
        return telemetry_df

    def create_error_features(self, errors):
        """
        Create features from error data that are predictive, not leaky.
        """
        print("🔄 Creating error features...")
        errors = errors.sort_values(['machineID', 'datetime'])
        error_dummies = pd.get_dummies(errors, columns=['errorID']).groupby(['machineID', 'datetime']).sum().reset_index()
        
        # Calculate time since last error for each type and cumulative count
        for col in [c for c in error_dummies.columns if 'errorID' in c]:
            error_dummies[f'time_since_{col}'] = error_dummies.groupby('machineID')['datetime'].diff().dt.total_seconds().fillna(0)
            error_dummies.loc[error_dummies[col] == 0, f'time_since_{col}'] = np.nan
            error_dummies[f'time_since_{col}'] = error_dummies.groupby('machineID')[f'time_since_{col}'].ffill().fillna(0)

            error_dummies[f'cumulative_{col}'] = error_dummies.groupby('machineID')[col].cumsum()
            
        print(f"✅ Error features created: {error_dummies.shape}")
        return error_dummies

    def create_maintenance_features(self, maint):
        """
        Create features from maintenance data that are predictive, not leaky.
        """
        print("🔄 Creating maintenance features...")
        maint = maint.sort_values(['machineID', 'datetime'])
        maint_dummies = pd.get_dummies(maint, columns=['comp']).groupby(['machineID', 'datetime']).sum().reset_index()

        for col in [c for c in maint_dummies.columns if 'comp_' in c]:
            maint_dummies[f'time_since_{col}'] = maint_dummies.groupby('machineID')['datetime'].diff().dt.total_seconds().fillna(0)
            maint_dummies.loc[maint_dummies[col] == 0, f'time_since_{col}'] = np.nan
            maint_dummies[f'time_since_{col}'] = maint_dummies.groupby('machineID')[f'time_since_{col}'].ffill().fillna(0)

        print(f"✅ Maintenance features created: {maint_dummies.shape}")
        return maint_dummies

    def create_failure_labels(self, base_df, failures_df):
        """
        Create failure labels for the next 24 hours.
        """
        print("🔄 Creating failure labels...")
        labeled_df = base_df.copy()
        labeled_df['failure'] = 0
        
        for _, failure in failures_df.iterrows():
            machine_id = failure['machineID']
            failure_time = failure['datetime']
            start_window = failure_time - timedelta(hours=24)
            
            mask = (
                (labeled_df['machineID'] == machine_id) &
                (labeled_df['datetime'] >= start_window) &
                (labeled_df['datetime'] < failure_time)
            )
            labeled_df.loc[mask, 'failure'] = 1
            
        print(f"✅ Failure labels created.")
        return labeled_df

    def create_sequences(self, df, target_col='failure'):
        """
        Create sequences for LSTM model.
        """
        print("🔄 Creating sequences for LSTM...")
        sequences, labels = [], []
        
        exclude_cols = ['datetime', 'machineID', 'model', target_col]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols and 'ID' not in col]
        
        df_values = df[self.feature_columns].values
        
        for i in range(len(df) - self.sequence_length):
            sequences.append(df_values[i:i + self.sequence_length])
            labels.append(df[target_col].iloc[i + self.sequence_length - 1])
            
        return np.array(sequences), np.array(labels)

    def process_all_data(self, data_dir='./'):
        """
        Process all PDM data and create features.
        """
        print("🚀 Starting PDM data processing...")
        data = self.load_data(data_dir)
        
        telemetry_features = self.create_telemetry_features(data['telemetry'])
        error_features = self.create_error_features(data['errors'])
        maint_features = self.create_maintenance_features(data['maint'])
        
        # Merge features
        final_data = pd.merge(telemetry_features, error_features, on=['datetime', 'machineID'], how='left')
        final_data = pd.merge(final_data, maint_features, on=['datetime', 'machineID'], how='left')
        final_data = pd.merge(final_data, data['machines'], on='machineID', how='left')
        
        # Fill NaNs from merges
        feature_cols = [col for col in final_data.columns if col not in ['datetime', 'machineID', 'model']]
        final_data[feature_cols] = final_data[feature_cols].fillna(0)
        
        # Encode model feature
        self.label_encoders['model'] = LabelEncoder()
        final_data['model'] = self.label_encoders['model'].fit_transform(final_data['model'])

        # Create labels
        labeled_data = self.create_failure_labels(final_data, data['failures'])
        
        # Split data before scaling
        X = labeled_data.drop(columns=['failure'])
        y = labeled_data['failure']
        
        # Create sequences
        sequences, labels = self.create_sequences(labeled_data)
        
        # Prepare tabular data
        X_tabular = labeled_data[self.feature_columns].copy()
        y_tabular = labeled_data['failure'].copy()
        
        # Scale features
        X_tabular_scaled = self.scaler.fit_transform(X_tabular)
        
        # Scale sequences
        num_features = sequences.shape[2]
        sequences_scaled = self.scaler.transform(sequences.reshape(-1, num_features)).reshape(sequences.shape)
        
        print(f"✅ Data processing completed!")
        return {
            'X_tabular': X_tabular_scaled,
            'y_tabular': y_tabular,
            'X_sequences': sequences_scaled,
            'y_sequences': labels,
            'feature_columns': self.feature_columns,
            'raw_data': labeled_data
        }

    def save_processor(self, filepath='pdm_processor.pkl'):
        """Save the processor for later use."""
        processor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length
        }
        joblib.dump(processor_data, filepath)
        print(f"✅ Processor saved to {filepath}")

if __name__ == '__main__':
    processor = PDMDataProcessor()
    processed_data = processor.process_all_data()
    processor.save_processor()
    
    print("\n📊 Data Summary:")
    print(f"   Tabular samples: {processed_data['X_tabular'].shape[0]}")
    print(f"   Sequence samples: {processed_data['X_sequences'].shape[0]}")
    print(f"   Features: {len(processed_data['feature_columns'])}")
    print(f"   Failure rate: {processed_data['y_tabular'].mean():.3f}")