-- Sample analysis queries

-- Average fuel economy by drivetrain
SELECT drive_wheels, AVG(city_mpg) AS avg_city_mpg, AVG(highway_mpg) AS avg_highway_mpg
FROM cars
GROUP BY drive_wheels
ORDER BY avg_highway_mpg DESC;

-- Heavier cars typically have lower MPG
SELECT
  CASE
    WHEN curb_weight < 2000 THEN '<2000'
    WHEN curb_weight < 2500 THEN '2000-2499'
    WHEN curb_weight < 3000 THEN '2500-2999'
    ELSE '3000+'
  END AS weight_bucket,
  AVG(highway_mpg) AS avg_highway_mpg
FROM cars
GROUP BY weight_bucket
ORDER BY avg_highway_mpg DESC;

-- MPG by body style
SELECT body_style, AVG(highway_mpg) AS avg_highway_mpg
FROM cars
GROUP BY body_style
ORDER BY avg_highway_mpg DESC;
