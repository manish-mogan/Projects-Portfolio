-- Data-Driven MPG (MySQL)
--
-- Usage:
-- 1) Create a database (example):
--    CREATE DATABASE IF NOT EXISTS cars_db;
--    USE cars_db;
-- 2) Run this file to create tables.

CREATE TABLE IF NOT EXISTS cars (
  car_id INT AUTO_INCREMENT PRIMARY KEY,
  make VARCHAR(64),
  fuel_type VARCHAR(32),
  aspiration VARCHAR(32),
  num_of_doors VARCHAR(16),
  body_style VARCHAR(32),
  drive_wheels VARCHAR(32),
  engine_location VARCHAR(32),

  wheel_base DOUBLE,
  length DOUBLE,
  width DOUBLE,
  height DOUBLE,
  curb_weight INT,

  engine_type VARCHAR(32),
  num_of_cylinders VARCHAR(32),
  engine_size INT,
  fuel_system VARCHAR(32),

  bore DOUBLE,
  stroke DOUBLE,
  compression_ratio DOUBLE,
  horsepower INT,
  peak_rpm INT,

  city_mpg INT,
  highway_mpg INT,
  price DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_cars_make ON cars (make);
CREATE INDEX IF NOT EXISTS idx_cars_mpg ON cars (highway_mpg, city_mpg);
CREATE INDEX IF NOT EXISTS idx_cars_weight ON cars (curb_weight);
