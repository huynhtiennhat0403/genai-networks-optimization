import pandas as pd
import numpy as np
import os

def balance_data(input_path="data/processed/train_data.csv",
                 output_path="data/processed/train_balanced.csv",
                 target_per_bin=80,
                 random_state=42):
    """
    Balance Resource Allocation (RA) distribution before training GAN.
    - Oversample bins with few samples using interpolation + small noise.
    - Undersample bins with too many samples.
    """

    np.random.seed(random_state)

    # Load processed train data
    df = pd.read_csv(input_path)

    # Xác định cột RA
    ra_col = "Resource Allocation"

    # Tìm unique levels (sau scaling)
    ra_values = df[ra_col].values
    unique_vals = sorted(np.unique(ra_values))
    print(f"Unique RA values before balancing: {unique_vals}")

    # Numeric: tất cả cột kiểu float/int, trừ one-hot (toàn 0/1)
    onehot_cols = [col for col in df.columns if col.startswith('Application Type_')]
    numeric_cols = [c for c in df.columns if c not in onehot_cols]
    print(f"Numeric columns: {numeric_cols}")
    balanced_parts = []

    for val in unique_vals:
        subset = df[df[ra_col] == val]

        if len(subset) < target_per_bin:
            # Oversample
            needed = target_per_bin - len(subset)
            new_samples = []

            for _ in range(needed):
                # chọn ngẫu nhiên 2 mẫu trong subset để nội suy
                rows = subset.sample(2, replace=True)
                s1_row = rows.iloc[0]
                s2_row = rows.iloc[1]
            
                alpha = np.random.uniform(0.2, 0.8)

                new_numeric = alpha * s1_row[numeric_cols] + (1 - alpha) * s2_row[numeric_cols]

                new_onehot = s1_row[onehot_cols]  # giữ nguyên one-hot từ một mẫu

                new_row = pd.concat([new_numeric, new_onehot])

                new_samples.append(new_row)

            new_df = pd.DataFrame(new_samples)
            balanced_parts.append(pd.concat([subset, new_df], axis=0))

        elif len(subset) > target_per_bin:
            # Undersample
            balanced_parts.append(subset.sample(target_per_bin, random_state=random_state))
        else:
            # Đủ target_per_bin rồi
            balanced_parts.append(subset)

    balanced_df = pd.concat(balanced_parts, axis=0).reset_index(drop=True)

    # Shuffle dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    balanced_df.to_csv(output_path, index=False)
    print(f"Balanced dataset saved to {output_path}")
    print(f"Balanced RA distribution:\n{balanced_df[ra_col].value_counts()}")

    return balanced_df