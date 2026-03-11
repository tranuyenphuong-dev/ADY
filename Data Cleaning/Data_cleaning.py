import pandas as pd
from sqlalchemy import create_engine

# 1. Connect to PostgreSQL
engine = create_engine("postgresql://postgres:Tduppostgresql1@localhost:5432/postgres")

# 2. Load data from PostgreSQL
df = pd.read_sql("SELECT * FROM mentalhealth", engine)

print("Original data shape:", df.shape)

# 3. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Remove duplicate rows
df = df.drop_duplicates()

#* 5. Fill missing values

# numeric columns → fill with mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
# df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean().round(1))

##! các cột giữ 1 chữ số thập phân
decimal_cols = ["study_hours_per_week", "gpa", "sleep_hours"]

for col in decimal_cols:
    df[col] = df[col].fillna(df[col].mean())
    df[col] = df[col].round(1)

#! các cột số làm tròn số nguyên
int_cols = [
    "credits",
    "exercise_frequency",
    "daily_caffeine_mg",
    "club_participation",
    "k6_item1",
    "k6_item2",
    "k6_item3",
    "k6_item4",
    "k6_item5",
    "k6_item6",
    "k6_total"
]

for col in int_cols:
    df[col] = df[col].fillna(df[col].mean())
    df[col] = df[col].round(0).astype(int)

# categorical columns → fill with "Unknown"
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

#* Check percentage of "Unknown" values in categorical columns
unknown_percentage = (df == "Unknown").sum() / len(df) * 100
print(unknown_percentage)

print(df['faculty'].mode())
print(df['academic_year'].mode())
print(df['housing_status'].mode())

df['faculty'] = df['faculty'].replace("Unknown", df['faculty'].mode()[0])
df['academic_year'] = df['academic_year'].replace("Unknown", df['academic_year'].mode()[0])
df['housing_status'] = df['housing_status'].replace("Unknown", df['housing_status'].mode()[0])

# 6. Basic data inspection
print("\nData info:")
print(df.info())

print("\nStatistics:")
print(df.describe())

# 7. Save cleaned data back to PostgreSQL
df.to_sql("mentalhealth_cleaned", engine, if_exists="replace", index=False)

print("\nCleaned data saved to table: mentalhealth_cleaned")
