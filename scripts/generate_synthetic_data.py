"""
Generate synthetic data to improve model performance in edge cases.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_data(base_df, n_synthetic=200):
    """Generate synthetic rows emphasizing problematic regions."""
    synthetic_rows = []
    np.random.seed(42)

    print(f"Original dataset: {len(base_df)} subjects")
    print(f"Good sleep (>=7): {len(base_df[base_df['quality_of_sleep'] >= 7])}")
    print(f"Poor sleep (<7): {len(base_df[base_df['quality_of_sleep'] < 7])}")

    # Oversleeping: stronger penalty
    print("\nGenerating oversleeping cases...")
    n_oversleep = 100
    for _ in range(n_oversleep):
        row = {
            'person_id': len(base_df) + len(synthetic_rows) + 1,
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(30, 55),
            'occupation': np.random.choice(['Teacher', 'Nurse', 'Salesperson', 'Doctor']),
            'sleep_duration': np.random.uniform(9.5, 12.0),
            'quality_of_sleep': np.random.choice([3, 4, 5, 6], p=[0.2, 0.4, 0.3, 0.1]),
            'physical_activity_level': np.random.randint(30, 70),
            'stress_level': np.random.randint(4, 9),
            'bmi_category': np.random.choice(['Normal', 'Overweight', 'Obese']),
            'blood_pressure': np.random.choice(['120/80', '125/82', '130/85', '140/90']),
            'heart_rate': np.random.randint(65, 90),
            'daily_steps': np.random.randint(4000, 9000),
            'sleep_disorder': np.random.choice(['None', 'Sleep Apnea', 'Insomnia'], p=[0.4, 0.4, 0.2]),
            'sleep_disorder_missing': 0
        }
        if row['sleep_disorder'] == 'None':
            row['sleep_disorder_missing'] = 1
        synthetic_rows.append(row)

    # Disorders
    print("Generating sleep disorder cases...")
    n_disorders = 50
    for _ in range(n_disorders):
        row = {
            'person_id': len(base_df) + len(synthetic_rows) + 1,
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(35, 60),
            'occupation': np.random.choice(['Nurse', 'Salesperson', 'Doctor', 'Sales Representative']),
            'sleep_duration': np.random.uniform(5.0, 7.5),
            'quality_of_sleep': np.random.choice([4, 5, 6]),
            'physical_activity_level': np.random.randint(20, 50),
            'stress_level': np.random.randint(6, 10),
            'bmi_category': np.random.choice(['Overweight', 'Obese', 'Normal']),
            'blood_pressure': np.random.choice(['130/85', '135/90', '140/90', '140/95']),
            'heart_rate': np.random.randint(75, 95),
            'daily_steps': np.random.randint(3000, 7000),
            'sleep_disorder': np.random.choice(['Sleep Apnea', 'Insomnia'], p=[0.6, 0.4]),
            'sleep_disorder_missing': 0
        }
        synthetic_rows.append(row)

    # Borderline 5–6.5h
    print("Generating borderline poor sleep cases...")
    n_borderline_poor = 60
    for _ in range(n_borderline_poor):
        row = {
            'person_id': len(base_df) + len(synthetic_rows) + 1,
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(28, 50),
            'occupation': np.random.choice(['Software Engineer', 'Doctor', 'Engineer', 'Scientist']),
            'sleep_duration': np.random.uniform(5.0, 6.5),
            'quality_of_sleep': np.random.choice([4, 5, 6], p=[0.3, 0.5, 0.2]),
            'physical_activity_level': np.random.randint(40, 70),
            'stress_level': np.random.randint(5, 8),
            'bmi_category': np.random.choice(['Normal', 'Normal Weight', 'Overweight']),
            'blood_pressure': np.random.choice(['115/75', '120/80', '125/80', '130/85']),
            'heart_rate': np.random.randint(70, 85),
            'daily_steps': np.random.randint(5000, 8000),
            'sleep_disorder': 'None',
            'sleep_disorder_missing': 1
        }
        synthetic_rows.append(row)

    # High stress
    print("Generating high stress cases...")
    n_high_stress = 60
    for _ in range(n_high_stress):
        row = {
            'person_id': len(base_df) + len(synthetic_rows) + 1,
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(30, 55),
            'occupation': np.random.choice(['Salesperson', 'Nurse', 'Sales Representative', 'Manager']),
            'sleep_duration': np.random.uniform(5.5, 7.0),
            'quality_of_sleep': np.random.choice([3, 4, 5, 6], p=[0.15, 0.45, 0.3, 0.1]),
            'physical_activity_level': np.random.randint(30, 60),
            'stress_level': np.random.randint(7, 10),
            'bmi_category': np.random.choice(['Normal', 'Overweight']),
            'blood_pressure': np.random.choice(['130/85', '135/90', '140/90']),
            'heart_rate': np.random.randint(78, 95),
            'daily_steps': np.random.randint(3500, 6500),
            'sleep_disorder': np.random.choice(['None', 'Insomnia'], p=[0.7, 0.3]),
            'sleep_disorder_missing': 1
        }
        if row['sleep_disorder'] == 'Insomnia':
            row['sleep_disorder_missing'] = 0
        synthetic_rows.append(row)

    # Excellent
    print("Generating excellent sleep profiles...")
    n_excellent = 30
    for _ in range(n_excellent):
        row = {
            'person_id': len(base_df) + len(synthetic_rows) + 1,
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(25, 50),
            'occupation': np.random.choice(['Accountant', 'Lawyer', 'Engineer', 'Doctor']),
            'sleep_duration': np.random.uniform(7.5, 8.5),
            'quality_of_sleep': np.random.choice([8, 9]),
            'physical_activity_level': np.random.randint(60, 90),
            'stress_level': np.random.randint(0, 4),
            'bmi_category': 'Normal',
            'blood_pressure': np.random.choice(['117/76', '120/80', '125/80']),
            'heart_rate': np.random.randint(60, 72),
            'daily_steps': np.random.randint(8000, 12000),
            'sleep_disorder': 'None',
            'sleep_disorder_missing': 1
        }
        synthetic_rows.append(row)

    # Critical short 4.0–5.5h
    print("Generating critical short sleep cases...")
    n_critical_short = 70
    for _ in range(n_critical_short):
        row = {
            'person_id': len(base_df) + len(synthetic_rows) + 1,
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(22, 55),
            'occupation': np.random.choice(['Salesperson', 'Engineer', 'Nurse', 'Other']),
            'sleep_duration': np.random.uniform(4.0, 5.5),
            'quality_of_sleep': np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2]),
            'physical_activity_level': np.random.randint(20, 60),
            'stress_level': np.random.randint(6, 10),
            'bmi_category': np.random.choice(['Normal', 'Overweight', 'Obese']),
            'blood_pressure': np.random.choice(['125/80', '130/85', '135/90', '140/90']),
            'heart_rate': np.random.randint(75, 100),
            'daily_steps': np.random.randint(2000, 8000),
            'sleep_disorder': np.random.choice(['None', 'Insomnia'], p=[0.6, 0.4]),
            'sleep_disorder_missing': 1
        }
        if row['sleep_disorder'] != 'None':
            row['sleep_disorder_missing'] = 0
        synthetic_rows.append(row)

    # Oversleepers with poor quality to reinforce penalties on >9h sleep
    print("Generating oversleep penalty cases...")
    n_penalty = max(1, n_synthetic // 5)
    for _ in range(n_penalty):
        disorder = np.random.choice(['None', 'Insomnia', 'Sleep Apnea'], p=[0.4, 0.35, 0.25])
        row = {
            'person_id': len(base_df) + len(synthetic_rows) + 1,
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(20, 65),
            'occupation': np.random.choice(['Student', 'Accountant', 'Doctor', 'Engineer', 'Nurse']),
            'sleep_duration': np.random.uniform(9.5, 12.0),
            'quality_of_sleep': np.random.choice([4, 5, 6], p=[0.45, 0.35, 0.20]),
            'physical_activity_level': np.random.randint(10, 40),
            'stress_level': np.random.randint(5, 9),
            'bmi_category': np.random.choice(['Overweight', 'Obese']),
            'blood_pressure': f"{np.random.randint(120, 150)}/{np.random.randint(80, 95)}",
            'heart_rate': np.random.randint(75, 95),
            'daily_steps': np.random.randint(1500, 6000),
            'sleep_disorder': disorder,
            'sleep_disorder_missing': 1
        }
        if disorder != 'None':
            row['sleep_disorder_missing'] = 0
        synthetic_rows.append(row)

    synthetic_df = pd.DataFrame(synthetic_rows)

    print(f"\nGenerated {len(synthetic_df)} synthetic samples")
    print(f"New total: {len(base_df) + len(synthetic_df)} subjects")
    print(f"Good sleep (>=7): {len(base_df[base_df['quality_of_sleep'] >= 7]) + len(synthetic_df[synthetic_df['quality_of_sleep'] >= 7])}")
    print(f"Poor sleep (<7): {len(base_df[base_df['quality_of_sleep'] < 7]) + len(synthetic_df[synthetic_df['quality_of_sleep'] < 7])}")

    return synthetic_df


def main():
    repo_root = Path(__file__).parent.parent
    original_df = pd.read_csv(repo_root / "reports" / "kaggle_clean_winsorized.csv")
    synthetic_df = generate_synthetic_data(original_df, n_synthetic=200)
    augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    output_path = repo_root / "reports" / "kaggle_augmented.csv"
    augmented_df.to_csv(output_path, index=False)

    print(f"\nAugmented dataset saved to: {output_path}")
    print(f"Original: {len(original_df)}, Synthetic: {len(synthetic_df)}, Total: {len(augmented_df)}")

    print("\n" + "="*60)
    print("DATASET AUGMENTATION SUMMARY")
    print("="*60)
    print(f"\nOriginal dataset: {len(original_df)} subjects")
    print(f"Synthetic data: {len(synthetic_df)} subjects")
    print(f"Total: {len(augmented_df)} subjects")

    print("\nQuality of Sleep Distribution:")
    print(augmented_df['quality_of_sleep'].value_counts().sort_index())

    print("\nGood vs Poor Sleep:")
    print(f"Good (>=7): {len(augmented_df[augmented_df['quality_of_sleep'] >= 7])}")
    print(f"Poor (<7): {len(augmented_df[augmented_df['quality_of_sleep'] < 7])}")

    print("\nOversleeping cases (>=9h):")
    oversleep = augmented_df[augmented_df['sleep_duration'] >= 9]
    print(f"Total: {len(oversleep)}")
    print(f"Good quality: {len(oversleep[oversleep['quality_of_sleep'] >= 7])}")
    print(f"Poor quality: {len(oversleep[oversleep['quality_of_sleep'] < 7])}")

    print("\nSleep disorder cases:")
    disorders = augmented_df[augmented_df['sleep_disorder'].isin(['Sleep Apnea', 'Insomnia'])]
    print(f"Total: {len(disorders)}")
    print(f"Good quality: {len(disorders[disorders['quality_of_sleep'] >= 7])}")
    print(f"Poor quality: {len(disorders[disorders['quality_of_sleep'] < 7])}")

    return augmented_df


if __name__ == "__main__":
    main()

