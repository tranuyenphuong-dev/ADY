SELECT * FROM public.mentalhealth

DROP TABLE IF EXISTS mentalhealth;

CREATE TABLE mentalhealth (
faculty TEXT,
academic_year TEXT,
credits NUMERIC,
study_hours_per_week NUMERIC,
gpa NUMERIC,
sleep_hours NUMERIC,
exercise_frequency NUMERIC,
daily_caffeine_mg NUMERIC,
housing_status TEXT,
club_participation NUMERIC,
k6_item1 NUMERIC,
k6_item2 NUMERIC,
k6_item3 NUMERIC,
k6_item4 NUMERIC,
k6_item5 NUMERIC,
k6_item6 NUMERIC,
k6_total NUMERIC
);

copy mentalhealth
FROM 'C:\Users\DELL\Downloads\MentalHealth.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',');

SELECT * 
FROM mentalhealth
LIMIT 400;