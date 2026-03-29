-- ─────────────────────────────────────────────────────────────────
-- CreditIQ — SQL Server database initialization
-- Run this once to create the database and application user.
--
-- How to run:
--   sqlcmd -S localhost,1433 -U sa -P "CreditIQ_Pass123!" -i scripts/init_db.sql
--
-- Or paste into VS Code SQLTools query window.
-- ─────────────────────────────────────────────────────────────────

-- Create the database
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'creditiq_db')
BEGIN
    CREATE DATABASE creditiq_db;
    PRINT 'Database creditiq_db created.';
END
ELSE
    PRINT 'Database creditiq_db already exists.';
GO

USE creditiq_db;
GO

-- Create application login (if using SQL auth, not Windows auth)
IF NOT EXISTS (SELECT name FROM sys.server_principals WHERE name = 'creditiq_user')
BEGIN
    CREATE LOGIN creditiq_user WITH PASSWORD = 'CreditIQ_User_Pass123!';
    PRINT 'Login creditiq_user created.';
END
GO

-- Create db user for the login
IF NOT EXISTS (SELECT name FROM sys.database_principals WHERE name = 'creditiq_user')
BEGIN
    CREATE USER creditiq_user FOR LOGIN creditiq_user;
    ALTER ROLE db_datareader ADD MEMBER creditiq_user;
    ALTER ROLE db_datawriter ADD MEMBER creditiq_user;
    ALTER ROLE db_ddladmin  ADD MEMBER creditiq_user;
    PRINT 'User creditiq_user created and roles assigned.';
END
GO

-- ─── Core Tables ───────────────────────────────────────────────

-- API users (for JWT authentication)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'api_users')
BEGIN
    CREATE TABLE api_users (
        id            INT IDENTITY(1,1) PRIMARY KEY,
        username      NVARCHAR(100) NOT NULL UNIQUE,
        email         NVARCHAR(255) NOT NULL UNIQUE,
        hashed_password NVARCHAR(255) NOT NULL,
        is_active     BIT DEFAULT 1,
        created_at    DATETIME2 DEFAULT GETUTCDATE(),
        updated_at    DATETIME2 DEFAULT GETUTCDATE()
    );
    PRINT 'Table api_users created.';
END
GO

-- Loan applications submitted through the API
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'loan_applications')
BEGIN
    CREATE TABLE loan_applications (
        id                  INT IDENTITY(1,1) PRIMARY KEY,
        application_ref     NVARCHAR(50) NOT NULL UNIQUE,  -- e.g. CIQ-2024-000001
        submitted_by        INT FOREIGN KEY REFERENCES api_users(id),
        -- Applicant details
        income              FLOAT,
        loan_amount         FLOAT,
        loan_term_months    INT,
        loan_purpose        NVARCHAR(100),
        employment_years    FLOAT,
        family_members      INT,
        education_type      NVARCHAR(100),
        income_type         NVARCHAR(100),
        -- Raw application JSON (full payload stored for audit)
        raw_payload         NVARCHAR(MAX),
        submitted_at        DATETIME2 DEFAULT GETUTCDATE()
    );
    PRINT 'Table loan_applications created.';
END
GO

-- Assessment results from the ML pipeline
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'assessment_results')
BEGIN
    CREATE TABLE assessment_results (
        id                      INT IDENTITY(1,1) PRIMARY KEY,
        application_id          INT NOT NULL FOREIGN KEY REFERENCES loan_applications(id),
        -- Scores
        credit_score            INT,           -- 300–850
        risk_band               NVARCHAR(20),  -- Low / Medium / High / Very High
        default_probability     FLOAT,         -- 0.0–1.0
        recommendation          NVARCHAR(50),  -- Approve / Review / Decline
        -- Sub-model outputs
        xgb_pd_score            FLOAT,
        sentiment_score         FLOAT,
        sentiment_label         NVARCHAR(20),
        lstm_forecast_json      NVARCHAR(MAX), -- JSON array: 12-month PD trajectory
        -- Explainability
        shap_values_json        NVARCHAR(MAX), -- JSON: top 10 SHAP features
        -- Metadata
        model_version           NVARCHAR(50),
        inference_time_ms       INT,
        assessed_at             DATETIME2 DEFAULT GETUTCDATE()
    );
    PRINT 'Table assessment_results created.';
END
GO

-- Model versions registry (mirrors MLflow but queryable via SQL)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'model_versions')
BEGIN
    CREATE TABLE model_versions (
        id              INT IDENTITY(1,1) PRIMARY KEY,
        model_name      NVARCHAR(100) NOT NULL,  -- xgboost / finbert / lstm / ensemble
        version         NVARCHAR(50)  NOT NULL,
        mlflow_run_id   NVARCHAR(255),
        roc_auc         FLOAT,
        ks_statistic    FLOAT,
        gini_coeff      FLOAT,
        is_active       BIT DEFAULT 0,
        trained_at      DATETIME2 DEFAULT GETUTCDATE(),
        notes           NVARCHAR(MAX)
    );
    PRINT 'Table model_versions created.';
END
GO

-- ─── Indexes ───────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_applications_ref
    ON loan_applications(application_ref);

CREATE INDEX IF NOT EXISTS idx_assessments_app
    ON assessment_results(application_id);

CREATE INDEX IF NOT EXISTS idx_assessments_date
    ON assessment_results(assessed_at);
GO

PRINT '─────────────────────────────────────────────────────────';
PRINT 'CreditIQ database initialized successfully.';
PRINT '─────────────────────────────────────────────────────────';
