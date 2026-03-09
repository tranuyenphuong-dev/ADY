DROP TABLE IF EXISTS MentalHealth;

CREATE TABLE MentalHealth (
faculty TEXT,
academic_year TEXT,
credits TEXT,
study_hours_per_week TEXT,
gpa TEXT,
sleep_hours TEXT,
exercise_frequency TEXT,
daily_caffeine_mg TEXT,
housing_status TEXT,
club_participation TEXT,
k6_item1 TEXT,
k6_item2 TEXT,
k6_item3 TEXT,
k6_item4 TEXT,
k6_item5 TEXT,
k6_item6 TEXT,
k6_total TEXT
);

SELECT column_name
FROM information_schema.columns
WHERE table_name = 'mentalhealth';

copy mentalhealth(
faculty,
academic_year,
credits,
study_hours_per_week,
gpa,
sleep_hours,
exercise_frequency,
daily_caffeine_mg,
housing_status,
club_participation,
k6_item1,
k6_item2,
k6_item3,
k6_item4,
k6_item5,
k6_item6,
k6_total
)
FROM 'C:\Users\DELL\Downloads\MentalHealth.csv'
WITH (
FORMAT csv,
HEADER true,
DELIMITER ','
);

SELECT *
FROM mentalhealth
LIMIT 400;

SELECT COUNT(*)
FROM mentalhealth;