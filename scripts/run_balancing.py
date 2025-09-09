from src.balancing import balance_data

if __name__ == "__main__":
    balance_data(input_path="data/processed/train_data.csv",
                 output_path="data/processed/train_balanced.csv",
                 target_per_bin=200,
                 random_state=42)