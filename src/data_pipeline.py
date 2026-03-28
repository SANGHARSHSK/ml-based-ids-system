import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pickle
import os

class DataPipeline:
    def __init__(self, filepath, sample_size=100000):
        self.filepath     = filepath
        self.sample_size  = sample_size
        self.scaler       = StandardScaler()
        self.features     = None

    # ──────────────────────────────
    # STEP 1 — LOAD DATA
    # ──────────────────────────────
    def load_data(self):
        print("\n📦 Loading data...")
        print(f"   File: {self.filepath}")

        df = pd.read_csv(
            self.filepath,
            nrows   = self.sample_size,
            encoding= 'utf-8',
            low_memory = False
        )

        # Strip spaces from column names
        df.columns = df.columns.str.strip()

        # Memory optimization for low RAM PC
        for col in df.select_dtypes('float64').columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes('int64').columns:
            df[col] = df[col].astype('int32')

        print(f"   ✅ Loaded shape : {df.shape}")
        print(f"   Columns        : {df.shape[1]}")
        print(f"   Rows           : {df.shape[0]}")
        return df

    # ──────────────────────────────
    # STEP 2 — CLEAN DATA
    # ──────────────────────────────
    def clean_data(self, df):
        print("\n🧹 Cleaning data...")

        # Drop non-useful columns
        drop_cols = [
            'Flow ID', 'Source IP', 'Destination IP',
            'Source Port', 'Timestamp', 'Fwd Header Length.1'
        ]
        df.drop(drop_cols, axis=1,
                errors='ignore', inplace=True)

        print(f"   Before cleaning : {df.shape}")

        # Replace infinity values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with NaN
        df.dropna(inplace=True)

        # Drop duplicate rows
        df.drop_duplicates(inplace=True)

        print(f"   After cleaning  : {df.shape}")
        return df

    # ──────────────────────────────
    # STEP 3 — ENCODE LABELS
    # ──────────────────────────────
    def encode_labels(self, df):
        print("\n🏷️  Encoding labels...")

        # Check what label column is called
        label_col = None
        for col in ['Label', 'label', ' Label']:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            print("   ❌ ERROR: No Label column found!")
            print("   Available columns:", df.columns.tolist())
            raise ValueError("Label column not found!")

        # Rename to standard name
        df.rename(columns={label_col: 'Label'}, inplace=True)

        # Show original labels
        print("   Original labels:")
        print(f"   {df['Label'].value_counts().to_dict()}")

        # Binary encode — BENIGN=0, any attack=1
        df['Label'] = df['Label'].apply(
            lambda x: 0 if str(x).strip().upper() == 'BENIGN'
                      else 1
        )

        normal = (df['Label'] == 0).sum()
        attack = (df['Label'] == 1).sum()
        total  = len(df)

        print(f"\n   ✅ Encoded labels:")
        print(f"   Normal (0) : {normal} ({normal/total*100:.1f}%)")
        print(f"   Attack (1) : {attack} ({attack/total*100:.1f}%)")

        return df

    # ──────────────────────────────
    # STEP 4 — BALANCE DATA
    # ──────────────────────────────
    def balance_data(self, X, y):
        print("\n⚖️  Balancing classes...")

        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)

        normal_idx = y_series[y_series == 0].index
        attack_idx = y_series[y_series == 1].index

        X_normal = X_df.loc[normal_idx]
        y_normal = y_series.loc[normal_idx]
        X_attack = X_df.loc[attack_idx]
        y_attack = y_series.loc[attack_idx]

        print(f"   Before → Normal: {len(X_normal)}, "
              f"Attack: {len(X_attack)}")

        # Upsample minority class to 50% of majority
        target_size = len(X_normal) // 2

        if len(X_attack) < target_size:
            X_attack_up, y_attack_up = resample(
                X_attack, y_attack,
                replace      = True,
                n_samples    = target_size,
                random_state = 42
            )
        else:
            X_attack_up = X_attack
            y_attack_up = y_attack

        # Combine balanced data
        X_balanced = np.vstack([
            X_normal.values,
            X_attack_up.values
            if hasattr(X_attack_up, 'values')
            else X_attack_up
        ])
        y_balanced = np.hstack([
            y_normal.values,
            y_attack_up.values
            if hasattr(y_attack_up, 'values')
            else y_attack_up
        ])

        # Shuffle
        shuffle_idx = np.random.permutation(len(y_balanced))
        X_balanced  = X_balanced[shuffle_idx]
        y_balanced  = y_balanced[shuffle_idx]

        print(f"   After  → Normal: {(y_balanced==0).sum()}, "
              f"Attack: {(y_balanced==1).sum()}")
        return X_balanced, y_balanced

    # ──────────────────────────────
    # STEP 5 — SCALE FEATURES
    # ──────────────────────────────
    def scale_features(self, X_train, X_test):
        print("\n📏 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)
        print("   ✅ Scaling done!")
        return X_train_scaled, X_test_scaled

    # ──────────────────────────────
    # STEP 6 — SAVE FILES
    # ──────────────────────────────
    def save_files(self, X_train, X_test,
                   y_train, y_test):
        print("\n💾 Saving files...")

        # Create directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models',         exist_ok=True)

        # Save numpy arrays
        np.save('data/processed/X_train.npy', X_train)
        np.save('data/processed/X_test.npy',  X_test)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/y_test.npy',  y_test)

        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save feature names
        with open('models/features.pkl', 'wb') as f:
            pickle.dump(self.features, f)

        print("   ✅ Saved:")
        print("   data/processed/X_train.npy")
        print("   data/processed/X_test.npy")
        print("   data/processed/y_train.npy")
        print("   data/processed/y_test.npy")
        print("   models/scaler.pkl")
        print("   models/features.pkl")

    # ──────────────────────────────
    # MAIN PIPELINE — RUN ALL STEPS
    # ──────────────────────────────
    def run(self):
        print("=" * 50)
        print("   🚀 STARTING DATA PIPELINE")
        print("=" * 50)

        # Step 1 — Load
        df = self.load_data()

        # Step 2 — Clean
        df = self.clean_data(df)

        # Step 3 — Encode labels
        df = self.encode_labels(df)

        # Step 4 — Split features and target
        print("\n✂️  Splitting features and target...")
        X = df.drop('Label', axis=1)
        y = df['Label'].values

        # Save feature names
        self.features = X.columns.tolist()
        print(f"   Total features : {len(self.features)}")

        # Step 5 — Train test split
        print("\n✂️  Train/Test split (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y,
            test_size    = 0.2,
            random_state = 42,
            stratify     = y
        )
        print(f"   Train size : {X_train.shape}")
        print(f"   Test size  : {X_test.shape}")

        # Step 6 — Balance ONLY training data
        X_train, y_train = self.balance_data(
            X_train, y_train)

        # Step 7 — Scale
        X_train, X_test = self.scale_features(
            X_train, X_test)

        # Step 8 — Save
        self.save_files(X_train, X_test,
                        y_train, y_test)

        print("\n" + "=" * 50)
        print("   ✅ PIPELINE COMPLETE!")
        print("=" * 50)
        print(f"\n   Final shapes:")
        print(f"   X_train : {X_train.shape}")
        print(f"   X_test  : {X_test.shape}")
        print(f"   y_train : {y_train.shape}")
        print(f"   y_test  : {y_test.shape}")

        return X_train, X_test, y_train, y_test


# ──────────────────────────────────────
# RUN PIPELINE
# ──────────────────────────────────────
if __name__ == "__main__":
    pipeline = DataPipeline(
        filepath    = 'data/network_traffic.csv',
        sample_size = 100000    # safe for 4-8GB RAM
    )
    pipeline.run()