"""Merge Sleep-EDF and Kaggle datasets"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"

def aggregate_subject_features(df):
    agg_features = []
    
    for subject in df['subject'].unique():
        subj_data = df[df['subject'] == subject].copy()
        n_epochs = len(subj_data)
        
        stage_counts = subj_data['stage'].value_counts()
        stage_pct = (stage_counts / n_epochs * 100).to_dict()
        total_time_hours = n_epochs * 30 / 3600
        
        sleep_stages = ['N1', 'N2', 'N3', 'REM']
        n_sleep_epochs = sum([stage_counts.get(s, 0) for s in sleep_stages])
        sleep_efficiency = (n_sleep_epochs / n_epochs * 100) if n_epochs > 0 else 0
        sleep_duration_hours = n_sleep_epochs * 30 / 3600
        
        first_sleep_idx = subj_data[subj_data['stage'].isin(sleep_stages)].index.min()
        if pd.notna(first_sleep_idx):
            subj_data_after_sleep = subj_data.loc[first_sleep_idx:]
            wake_after_sleep = subj_data_after_sleep[subj_data_after_sleep['stage'] == 'W'].shape[0]
            waso_minutes = wake_after_sleep * 30 / 60
            first_sleep_epoch = subj_data[subj_data['stage'].isin(sleep_stages)]['epoch'].min()
            sleep_latency_minutes = first_sleep_epoch * 30 / 60 if pd.notna(first_sleep_epoch) else 0
        else:
            waso_minutes = 0
            sleep_latency_minutes = 0
        
        eeg_cols = ['eeg_delta_rel', 'eeg_theta_rel', 'eeg_alpha_rel', 'eeg_beta_rel']
        other_cols = ['eog_var', 'emg_rms']
        
        feature_dict = {
            'subject': subject,
            'n_epochs': n_epochs,
            'total_time_hours': total_time_hours,
            'sleep_duration_hours': sleep_duration_hours,
            'sleep_efficiency_pct': sleep_efficiency,
            'sleep_latency_min': sleep_latency_minutes,
            'waso_min': waso_minutes,
            'stage_W_pct': stage_pct.get('W', 0),
            'stage_N1_pct': stage_pct.get('N1', 0),
            'stage_N2_pct': stage_pct.get('N2', 0),
            'stage_N3_pct': stage_pct.get('N3', 0),
            'stage_REM_pct': stage_pct.get('REM', 0),
        }
        
        for col in eeg_cols + other_cols:
            feature_dict[f'{col}_mean'] = subj_data[col].mean()
            feature_dict[f'{col}_std'] = subj_data[col].std()
            feature_dict[f'{col}_median'] = subj_data[col].median()
        
        agg_features.append(feature_dict)
    
    return pd.DataFrame(agg_features)

def create_synthetic_sleep_quality(row):
    sleep_eff = row['sleep_efficiency_pct']
    n3_pct = row['stage_N3_pct']
    rem_pct = row['stage_REM_pct']
    
    if sleep_eff >= 85 and n3_pct >= 15 and rem_pct >= 20:
        return 8
    elif sleep_eff >= 75 and (n3_pct >= 10 or rem_pct >= 15):
        return 7
    elif sleep_eff >= 65:
        return 6
    else:
        return 5

if __name__ == "__main__":
    print("Merging datasets...")
    
    edf_raw = pd.read_csv(REPORTS / "sleepedf_features.csv")
    edf_aggregated = aggregate_subject_features(edf_raw)
    edf_aggregated['quality_of_sleep'] = edf_aggregated.apply(create_synthetic_sleep_quality, axis=1)
    
    kaggle = pd.read_csv(REPORTS / "kaggle_clean_winsorized.csv")
    
    proxy_values = {
        'age': int(kaggle['age'].median()),
        'gender': kaggle['gender'].mode()[0],
        'occupation': 'Other',
        'physical_activity_level': int(kaggle['physical_activity_level'].median()),
        'stress_level': int(kaggle['stress_level'].median()),
        'bmi_category': kaggle['bmi_category'].mode()[0],
        'blood_pressure': kaggle['blood_pressure'].mode()[0],
        'heart_rate': int(kaggle['heart_rate'].median()),
        'daily_steps': int(kaggle['daily_steps'].median()),
        'sleep_disorder': None,
        'sleep_disorder_missing': 1,
    }
    
    edf_with_lifestyle = edf_aggregated.copy()
    edf_with_lifestyle['sleep_duration'] = edf_with_lifestyle['sleep_duration_hours']
    for col, value in proxy_values.items():
        edf_with_lifestyle[col] = value
    
    max_person_id = kaggle['person_id'].max()
    edf_with_lifestyle['person_id'] = range(max_person_id + 1, max_person_id + 1 + len(edf_with_lifestyle))
    
    edf_with_lifestyle.to_csv(REPORTS / "sleepedf_subject_aggregated.csv", index=False)
    
    kaggle_cols = ['person_id', 'gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep',
                   'physical_activity_level', 'stress_level', 'bmi_category', 'blood_pressure',
                   'heart_rate', 'daily_steps', 'sleep_disorder', 'sleep_disorder_missing']
    
    edf_specific_cols = [col for col in edf_with_lifestyle.columns 
                          if col not in kaggle_cols and col != 'subject']
    
    kaggle_extended = kaggle.copy()
    for col in edf_specific_cols:
        kaggle_extended[col] = np.nan
    kaggle_extended['subject'] = np.nan
    
    edf_for_merge = edf_with_lifestyle[kaggle_extended.columns].copy()
    merged_data = pd.concat([kaggle_extended, edf_for_merge], axis=0, ignore_index=True)
    
    merged_data.to_csv(REPORTS / "kaggle_edf_merged.csv", index=False)
    
    print(f"Complete: {merged_data.shape[0]} subjects, {merged_data.shape[1]} features")
