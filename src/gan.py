import pandas as pd
from sdv.single_table.ctgan import CTGAN

def train_ctgan(train_path, model_save_path):
    # 1. Load data
    df = pd.read_csv(train_path)

    # 2. Xác định các cột phân loại và liên tục
    numerical_columns = [col for col in df.columns if col != 'Application Type']

    # 3. Kiểm tra dữ liệu đầu vào
    print("Checking input data ranges:")
    for col in numerical_columns:
        min_val, max_val = df[col].min(), df[col].max()
        print(f"{col}: min={min_val:.3f}, max={max_val:.3f}")
    print(f"Application Type unique values: {sorted(df['Application Type'].unique())}")

    # 4. Tạo metadata (tùy chọn - SDV có thể auto-detect)
    # metadata = SingleTableMetadata()
    # metadata.detect_from_dataframe(df)
    # 
    # # Cập nhật metadata để chỉ định rõ kiểu dữ liệu
    # for col in numerical_columns:
    #     metadata.update_column(column_name=col, sdtype='numerical')
    # metadata.update_column(column_name='Application Type', sdtype='categorical')

    # 5. Không sử dụng constraints phức tạp, chỉ đảm bảo data clean
    # Vì Inequality constraint cần tạo thêm columns, ta sẽ dùng cách đơn giản hơn
    constraints = []
    
    # Đảm bảo training data clean trước
    print("Ensuring all numerical columns are in [0, 1] range...")
    for col in numerical_columns:
        df[col] = df[col].clip(0, 1)
        print(f"Clipped {col} to [0, 1] range")

    # 6. Train CTGAN
    model = CTGAN(verbose=True, batch_size=50, epochs=100)
    
    # Fit model - SDV sẽ tự detect metadata từ dataframe
    model.fit(df)

    # 7. Save model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

def generate_synthetic(model_path, num_samples, output_path):
    # Load model
    model = CTGAN.load(model_path)

    # Generate new synthetic samples
    synthetic_data = model.sample(num_samples)

    # Post-process: Đảm bảo Application Type là số nguyên từ 0 đến 10
    if 'Application Type' in synthetic_data.columns:
        synthetic_data['Application Type'] = synthetic_data['Application Type'].round().astype(int)
        synthetic_data['Application Type'] = synthetic_data['Application Type'].clip(0, 10)

    # Post-process: Đảm bảo các cột số nằm trong [0, 1] (quan trọng!)
    numerical_columns = [col for col in synthetic_data.columns if col != 'Application Type']
    for col in numerical_columns:
        synthetic_data[col] = synthetic_data[col].clip(0, 1)

    # Kiểm tra dữ liệu tổng hợp
    print("Checking synthetic data ranges:")
    for col in numerical_columns:
        min_val, max_val = synthetic_data[col].min(), synthetic_data[col].max()
        print(f"{col}: min={min_val:.3f}, max={max_val:.3f}")
    if 'Application Type' in synthetic_data.columns:
        print(f"Application Type unique values: {sorted(synthetic_data['Application Type'].unique())}")

    # Save
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")

def validate_constraints(df):
    """Kiểm tra xem dữ liệu có thỏa mãn constraints không"""
    print("\nValidating constraints:")
    numerical_columns = [col for col in df.columns if col != 'Application Type']
    
    for col in numerical_columns:
        min_val, max_val = df[col].min(), df[col].max()
        if min_val < 0 or max_val > 1:
            print(f"⚠️  {col}: OUT OF RANGE! min={min_val:.3f}, max={max_val:.3f}")
        else:
            print(f"✅ {col}: OK, min={min_val:.3f}, max={max_val:.3f}")
    
    if 'Application Type' in df.columns:
        unique_vals = sorted(df['Application Type'].unique())
        print(f"Application Type values: {unique_vals}")

if __name__ == "__main__":
    train_csv = "data/processed/train_data.csv"
    model_path = "models/ctgan.pkl"
    synthetic_path = "data/synthetic/synthetic_data.csv"

    # Train model
    print("=== TRAINING CTGAN ===")
    train_ctgan(train_csv, model_path)

    # Generate synthetic data
    print("\n=== GENERATING SYNTHETIC DATA ===")
    generate_synthetic(model_path, num_samples=2000, output_path=synthetic_path)
    
    # Validate generated data
    print("\n=== VALIDATING SYNTHETIC DATA ===")
    synthetic_df = pd.read_csv(synthetic_path)
    validate_constraints(synthetic_df)