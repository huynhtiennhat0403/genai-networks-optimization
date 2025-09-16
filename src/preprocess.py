import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import resample
import joblib
import os

def parse_bandwidth(bandwidth_str):
    if 'Mbps' in bandwidth_str:
        return float(bandwidth_str.replace(' Mbps', ''))
    elif 'Kbps' in bandwidth_str:
        return float(bandwidth_str.replace(' Kbps', '')) / 1000
    else:
        return float(bandwidth_str)
    
def balance_data(df, target_col='Resource Allocation', target_values=[50,55,60,65,70,75,80,85,90], target_size=70, random_state=42):
    balanced_frames = []
    for val in target_values:
        subset = df[df[target_col] == val]

        if len(subset) == 0:
            continue  # bỏ qua nếu không có mẫu nào

        if val == 70:
            # undersample nếu nhiều hơn 70
            subset_balanced = resample(
                subset,
                replace=False,
                n_samples=min(target_size, len(subset)),
                random_state=random_state
            )
        else:
            # oversample nếu ít hơn 70
            subset_balanced = resample(
                subset,
                replace=True,
                n_samples=target_size,
                random_state=random_state
            )

        balanced_frames.append(subset_balanced)

    df_balanced = pd.concat(balanced_frames).reset_index(drop=True)
    return df_balanced


def preprocess_data(input_path, output_path="data/processed/", models_path="models/", test_size=0.2):
    # Load data
    df = pd.read_csv(input_path)

    df.columns = df.columns.str.replace('_', ' ')
    df['Application Type'] = df['Application Type'].str.replace('_', ' ')

    # Drop unnecessary columns
    df = df.drop(columns=["Timestamp", "User ID"])

    # Convert object columns to numerical
    df['Signal Strength'] = df['Signal Strength'].str.replace(' dBm', '').astype(float)
    df['Latency'] = df['Latency'].str.replace(' ms', '').astype(float)
    df['Resource Allocation'] = df['Resource Allocation'].str.replace('%', '').astype(float)
    df['Required Bandwidth'] = df['Required Bandwidth'].apply(parse_bandwidth)
    df['Allocated Bandwidth'] = df['Allocated Bandwidth'].apply(parse_bandwidth)

    # Encode application type using LabelEncoder
    encoder = LabelEncoder()
    df['Application Type'] = encoder.fit_transform(df['Application Type'])

    # Balance data
    df['Resource Allocation'] = df['Resource Allocation'].astype(int)
    df = balance_data(df, target_col='Resource Allocation')
    print(f"Data shape after balancing: {df.shape}")

    # Save the encoder
    os.makedirs(models_path, exist_ok=True)
    encoder_path = os.path.join(models_path, 'encoder.pkl')
    joblib.dump(encoder, encoder_path)  
    print(f"Encoder saved to {encoder_path}")

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Scale numerical features (excluding Application Type)
    numeric_features = [col for col in df.columns if not col.startswith("Application")]
    scaler = MinMaxScaler()
    train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
    test_df[numeric_features] = scaler.transform(test_df[numeric_features])

    # Save the scaler
    scaler_path = os.path.join(models_path, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    train_path = os.path.join(output_path, 'train_data.csv')
    test_path = os.path.join(output_path, 'test_data.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Processed training data saved to {train_path}")
    print(f"Processed testing data saved to {test_path}")
    
    return train_df, test_df


def encode_application_type(app_type_str, models_path="models/"):
    encoder_path = os.path.join(models_path, 'encoder.pkl')
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at {encoder_path}. Please run preprocess_data first.")
    encoder = joblib.load(encoder_path)
    app_type_encoded = encoder.transform([app_type_str])
    return app_type_encoded[0]

if __name__ == '__main__':
    preprocess_data(input_path="data/raw/Quality of Service 5G.csv")