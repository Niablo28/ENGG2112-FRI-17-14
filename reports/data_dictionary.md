# Data Dictionary

- Total rows: **374**
- Total columns: **14**
- Target column: **quality_of_sleep**

## Summary Table
(详见 CSV：`reports/data_dictionary.csv`)

| column | dtype | missing_% | n_unique | examples | min | max | mean | std |
|---|---|---:|---:|---|---:|---:|---:|---:|
| person_id | int64 | 0.0 | 374 | 1, 2, 3 |  |  |  |  |
| gender | object | 0.0 | 2 | Male, Male, Male |  |  |  |  |
| age | int64 | 0.0 | 31 | 27, 28, 28 | 27 | 59 | 42.1845 | 8.6731 |
| occupation | object | 0.0 | 11 | Software Engineer, Doctor, Doctor |  |  |  |  |
| sleep_duration | float64 | 0.0 | 27 | 6.1, 6.2, 6.2 | 5.8 | 8.5 | 7.1321 | 0.7957 |
| quality_of_sleep | int64 | 0.0 | 6 | 6, 6, 6 |  |  |  |  |
| physical_activity_level | int64 | 0.0 | 16 | 42, 60, 60 | 30 | 90 | 59.1711 | 20.8308 |
| stress_level | int64 | 0.0 | 6 | 6, 8, 8 | 3 | 8 | 5.385 | 1.7745 |
| bmi_category | object | 0.0 | 4 | Overweight, Normal, Normal |  |  |  |  |
| blood_pressure | object | 0.0 | 25 | 126/83, 125/80, 125/80 |  |  |  |  |
| heart_rate | int64 | 0.0 | 12 | 77, 75, 75 | 65 | 78 | 69.9652 | 3.5673 |
| daily_steps | int64 | 0.0 | 20 | 4200, 10000, 10000 | 3000 | 10000 | 6816.8449 | 1617.9157 |
| sleep_disorder | object | 58.56 | 2 | Sleep Apnea, Sleep Apnea, Insomnia |  |  |  |  |
| sleep_disorder_missing | int64 | 0.0 | 2 | 1, 1, 1 | 0 | 1 | 0.5856 | 0.4933 |

## Categorical Levels (Top-10)
- **gender**: Male(189), Female(185)
- **occupation**: Nurse(73), Doctor(71), Engineer(63), Lawyer(47), Teacher(40), Accountant(37), Salesperson(32), Software Engineer(4), Scientist(4), Sales Representative(2)
- **bmi_category**: Normal(195), Overweight(148), Normal Weight(21), Obese(10)
- **blood_pressure**: 130/85(99), 140/95(65), 125/80(65), 120/80(45), 115/75(32), 135/90(27), 140/90(4), 125/82(4), 132/87(3), 128/85(3)
- **sleep_disorder**: Sleep Apnea(78), Insomnia(77)
