-- =============================================================================
-- PostgreSQL Database Initialization Script
-- Medical Claims Data Engineering Platform
-- =============================================================================

-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS intermediate;
CREATE SCHEMA IF NOT EXISTS marts;
CREATE SCHEMA IF NOT EXISTS metadata;

-- =============================================================================
-- RAW SCHEMA - Landing zone tables
-- =============================================================================

-- Raw claims table
CREATE TABLE IF NOT EXISTS raw.claims_raw (
    claim_id_key BIGINT,
    member_state VARCHAR(10),
    member_county INTEGER,
    age VARCHAR(10),
    sex CHAR(1),
    form_type CHAR(1),
    sv_stat VARCHAR(10),
    product_type VARCHAR(10),
    pos VARCHAR(10),
    icd_diag_01 VARCHAR(20),
    icd_diag_02 VARCHAR(20),
    cpt VARCHAR(10),
    amt_billed DECIMAL(15,2),
    amt_paid DECIMAL(15,2),
    amt_deduct DECIMAL(15,2),
    amt_coins DECIMAL(15,2),
    client_los INTEGER,
    qty INTEGER,
    service_date DATE,
    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    _source_file VARCHAR(255)
);

-- Create index on claim_id_key
CREATE INDEX IF NOT EXISTS idx_claims_raw_claim_id ON raw.claims_raw(claim_id_key);
CREATE INDEX IF NOT EXISTS idx_claims_raw_service_date ON raw.claims_raw(service_date);

-- =============================================================================
-- STAGING SCHEMA - Validated and cleaned data
-- =============================================================================

-- Staged claims view (mirror of DBT stg_claims)
CREATE OR REPLACE VIEW staging.stg_claims AS
SELECT
    claim_id_key,
    member_state,
    member_county,
    age,
    sex AS gender_code,
    form_type,
    sv_stat AS service_status,
    product_type,
    pos AS place_of_service,
    icd_diag_01 AS primary_diagnosis,
    icd_diag_02 AS secondary_diagnosis,
    cpt AS procedure_code,
    COALESCE(amt_billed, 0) AS amt_billed,
    COALESCE(amt_paid, 0) AS amt_paid,
    COALESCE(amt_deduct, 0) AS amt_deductible,
    COALESCE(amt_coins, 0) AS amt_coinsurance,
    COALESCE(client_los, 0) AS length_of_stay,
    COALESCE(qty, 1) AS quantity,
    service_date,
    _loaded_at
FROM raw.claims_raw
WHERE claim_id_key IS NOT NULL
  AND amt_billed >= 0
  AND amt_paid >= 0;

-- =============================================================================
-- MARTS SCHEMA - Analytics-ready tables
-- =============================================================================

-- Dimension: Patient
CREATE TABLE IF NOT EXISTS marts.dim_patient (
    patient_key SERIAL PRIMARY KEY,
    member_state VARCHAR(10),
    member_county INTEGER,
    age_band VARCHAR(20),
    gender_code CHAR(1),
    gender_description VARCHAR(20),
    residency_type VARCHAR(20),
    valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_to TIMESTAMP,
    is_current BOOLEAN DEFAULT TRUE
);

-- Dimension: Diagnosis
CREATE TABLE IF NOT EXISTS marts.dim_diagnosis (
    diagnosis_key SERIAL PRIMARY KEY,
    icd_code VARCHAR(20),
    icd_category CHAR(1),
    category_description VARCHAR(100),
    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Procedure
CREATE TABLE IF NOT EXISTS marts.dim_procedure (
    procedure_key SERIAL PRIMARY KEY,
    cpt_code VARCHAR(10),
    procedure_category VARCHAR(50),
    description VARCHAR(255),
    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Date
CREATE TABLE IF NOT EXISTS marts.dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE,
    year INTEGER,
    quarter INTEGER,
    month INTEGER,
    month_name VARCHAR(20),
    week INTEGER,
    day_of_month INTEGER,
    day_of_week INTEGER,
    day_name VARCHAR(20),
    is_weekend BOOLEAN
);

-- Fact: Claims
CREATE TABLE IF NOT EXISTS marts.fact_claims (
    claim_fact_key VARCHAR(64) PRIMARY KEY,
    claim_id_key BIGINT,
    patient_key INTEGER REFERENCES marts.dim_patient(patient_key),
    primary_diagnosis_key INTEGER REFERENCES marts.dim_diagnosis(diagnosis_key),
    form_type CHAR(1),
    service_status VARCHAR(10),
    product_type VARCHAR(10),
    place_of_service VARCHAR(10),
    procedure_code VARCHAR(10),
    amt_billed DECIMAL(15,2),
    amt_paid DECIMAL(15,2),
    amt_deductible DECIMAL(15,2),
    amt_coinsurance DECIMAL(15,2),
    out_of_pocket DECIMAL(15,2),
    net_amount DECIMAL(15,2),
    length_of_stay INTEGER,
    quantity INTEGER,
    payment_ratio DECIMAL(8,4),
    is_high_cost BOOLEAN,
    num_diagnoses INTEGER,
    service_date DATE,
    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes on fact table
CREATE INDEX IF NOT EXISTS idx_fact_claims_patient ON marts.fact_claims(patient_key);
CREATE INDEX IF NOT EXISTS idx_fact_claims_diagnosis ON marts.fact_claims(primary_diagnosis_key);
CREATE INDEX IF NOT EXISTS idx_fact_claims_date ON marts.fact_claims(service_date);

-- =============================================================================
-- METADATA SCHEMA - Pipeline tracking
-- =============================================================================

-- Pipeline execution history
CREATE TABLE IF NOT EXISTS metadata.pipeline_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    pipeline_name VARCHAR(100),
    status VARCHAR(20),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds DECIMAL(10,2),
    rows_processed BIGINT,
    rows_failed BIGINT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data quality results
CREATE TABLE IF NOT EXISTS metadata.quality_checks (
    check_id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) REFERENCES metadata.pipeline_runs(run_id),
    table_name VARCHAR(100),
    check_type VARCHAR(50),
    column_name VARCHAR(100),
    expectation VARCHAR(255),
    success BOOLEAN,
    observed_value TEXT,
    expected_value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data lineage
CREATE TABLE IF NOT EXISTS metadata.lineage (
    lineage_id SERIAL PRIMARY KEY,
    source_id VARCHAR(100),
    target_id VARCHAR(100),
    operation VARCHAR(50),
    columns_affected TEXT[],
    transformation_logic TEXT,
    job_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data catalog
CREATE TABLE IF NOT EXISTS metadata.data_catalog (
    asset_id VARCHAR(100) PRIMARY KEY,
    asset_name VARCHAR(100),
    asset_type VARCHAR(20),
    layer VARCHAR(20),
    location VARCHAR(500),
    format VARCHAR(20),
    description TEXT,
    owner VARCHAR(100),
    tags TEXT[],
    row_count BIGINT,
    size_bytes BIGINT,
    quality_score DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Populate dim_date (for 2016)
-- =============================================================================

INSERT INTO marts.dim_date (date_key, full_date, year, quarter, month, month_name, week, day_of_month, day_of_week, day_name, is_weekend)
SELECT
    TO_CHAR(d, 'YYYYMMDD')::INTEGER AS date_key,
    d AS full_date,
    EXTRACT(YEAR FROM d)::INTEGER AS year,
    EXTRACT(QUARTER FROM d)::INTEGER AS quarter,
    EXTRACT(MONTH FROM d)::INTEGER AS month,
    TO_CHAR(d, 'Month') AS month_name,
    EXTRACT(WEEK FROM d)::INTEGER AS week,
    EXTRACT(DAY FROM d)::INTEGER AS day_of_month,
    EXTRACT(DOW FROM d)::INTEGER AS day_of_week,
    TO_CHAR(d, 'Day') AS day_name,
    EXTRACT(DOW FROM d) IN (0, 6) AS is_weekend
FROM generate_series('2016-01-01'::DATE, '2016-12-31'::DATE, '1 day'::INTERVAL) AS d
ON CONFLICT (date_key) DO NOTHING;

-- =============================================================================
-- Create helpful views
-- =============================================================================

-- Claims summary by month
CREATE OR REPLACE VIEW marts.vw_claims_monthly_summary AS
SELECT
    DATE_TRUNC('month', service_date) AS month,
    COUNT(*) AS claim_count,
    SUM(amt_billed) AS total_billed,
    SUM(amt_paid) AS total_paid,
    AVG(amt_paid) AS avg_paid,
    AVG(payment_ratio) AS avg_payment_ratio
FROM marts.fact_claims
GROUP BY DATE_TRUNC('month', service_date)
ORDER BY month;

-- Claims summary by age band
CREATE OR REPLACE VIEW marts.vw_claims_by_age AS
SELECT
    p.age_band,
    COUNT(*) AS claim_count,
    SUM(f.amt_paid) AS total_paid,
    AVG(f.amt_paid) AS avg_paid,
    AVG(f.payment_ratio) AS avg_payment_ratio
FROM marts.fact_claims f
JOIN marts.dim_patient p ON f.patient_key = p.patient_key
GROUP BY p.age_band
ORDER BY p.age_band;

-- Grant permissions
GRANT USAGE ON SCHEMA raw TO PUBLIC;
GRANT USAGE ON SCHEMA staging TO PUBLIC;
GRANT USAGE ON SCHEMA marts TO PUBLIC;
GRANT USAGE ON SCHEMA metadata TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA raw TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA staging TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA marts TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA metadata TO PUBLIC;

-- Log initialization
INSERT INTO metadata.pipeline_runs (run_id, pipeline_name, status, start_time, end_time, rows_processed)
VALUES ('init-db-001', 'database_initialization', 'success', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0);

COMMIT;
