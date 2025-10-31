"""
Check for duplicate leakage across train/test splits.
Sanity check: ensure no subject appears in both splits (by person_id if available),
and check for exact duplicate rows across splits.
"""
import pandas as pd
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def main():
    print("Checking for duplicate leakage across train/test splits...\n")
    
    # Load preprocessed splits
    try:
        Xtr = pd.read_parquet(REPO / "reports" / "X_train_proc.parquet")
        Xte = pd.read_parquet(REPO / "reports" / "X_test_proc.parquet")
        ytr = pd.read_csv(REPO / "reports" / "y_train.csv").squeeze()
        yte = pd.read_csv(REPO / "reports" / "y_test.csv").squeeze()
        
        print(f"Train set: {len(Xtr)} samples")
        print(f"Test set: {len(Xte)} samples")
        
        # Check for exact duplicate rows across splits
        all_data = pd.concat([Xtr, Xte], ignore_index=True)
        dup_count = all_data.duplicated().sum()
        print(f"\nExact duplicate rows across splits: {dup_count}")
        
        if dup_count > 0:
            print("⚠️  WARNING: Duplicate rows found! This may indicate data leakage.")
            print("   Check for duplicate person_ids or identical feature combinations.")
        else:
            print("✓ No exact duplicates found.")
            
    except FileNotFoundError as e:
        print(f"⚠️  Could not find preprocessed splits: {e}")
        print("   This check requires preprocessed train/test files.")
    
    # Check raw data for person_id overlap if available
    try:
        raw = pd.read_csv(REPO / "reports" / "kaggle_augmented.csv")
        if "person_id" in raw.columns:
            print("\nChecking person_id overlap in raw data...")
            # Try to reconstruct splits based on typical 80/20 split
            # This is approximate - actual split logic may differ
            n_train = int(len(raw) * 0.8)
            raw_tr = raw.iloc[:n_train]
            raw_te = raw.iloc[n_train:]
            
            overlap = set(raw_tr["person_id"]) & set(raw_te["person_id"])
            if overlap:
                print(f"⚠️  WARNING: {len(overlap)} person_ids appear in both train and test (approximate check).")
                print(f"   Overlapping IDs: {list(overlap)[:10]}...")
            else:
                print("✓ No person_id overlap detected (approximate check).")
    except FileNotFoundError:
        print("\n⚠️  Could not check person_id overlap: raw data file not found.")
    except KeyError:
        print("\n⚠️  Raw data does not contain 'person_id' column.")
    
    print("\n" + "="*60)
    print("RECOMMENDATION: Ensure proper train/test split using person_id or stable key")
    print("to prevent subject-level leakage. Use stratified split by person_id if available.")


if __name__ == "__main__":
    main()

