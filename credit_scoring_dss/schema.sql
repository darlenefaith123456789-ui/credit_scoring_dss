-- Credit Scoring Decision Support System: MySQL tables
-- app.py runs this file automatically on start-up (CREATE ... IF NOT EXISTS).
-- To run it by hand:  mysql -u root -p credit_scoring_dss < schema.sql

-- System users (loan officers and administrators). Passwords are stored as hashes.
CREATE TABLE IF NOT EXISTS users (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    username        VARCHAR(50) NOT NULL UNIQUE,
    full_name       VARCHAR(120) NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    role            ENUM('admin','officer') NOT NULL DEFAULT 'officer',
    status          ENUM('active','pending','disabled') NOT NULL DEFAULT 'active',  -- pending = requested on the sign-in page
    last_login      TIMESTAMP NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Borrower profile entered by the loan officer
CREATE TABLE IF NOT EXISTS applicants (
    id                     INT AUTO_INCREMENT PRIMARY KEY,
    applicant_name         VARCHAR(120) NULL,
    age                    TINYINT UNSIGNED NOT NULL,
    marital_status         TINYINT UNSIGNED NOT NULL,      -- code, see model labels
    employment_type        TINYINT UNSIGNED NOT NULL,      -- code, see model labels
    monthly_income         DECIMAL(12,2) NOT NULL,
    current_debt           DECIMAL(12,2) NOT NULL DEFAULT 0,
    missed_payments_count  SMALLINT UNSIGNED NOT NULL DEFAULT 0,
    late_payment_penalty   DECIMAL(12,2) NOT NULL DEFAULT 0,
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Loan applied for and the system's assessment of it
CREATE TABLE IF NOT EXISTS loan_applications (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    applicant_id         INT NOT NULL,
    assessed_by          INT NULL,                         -- users.id of the loan officer
    product_id           VARCHAR(40) NULL,                 -- motorcycle from products.json
    product_name         VARCHAR(120) NULL,
    loan_product         TINYINT UNSIGNED NOT NULL,
    credit_amount        DECIMAL(12,2) NOT NULL,
    term_months          SMALLINT UNSIGNED NOT NULL,
    interest_rate_pa     DECIMAL(5,2) NOT NULL,
    computation_mode     TINYINT UNSIGNED NOT NULL,
    monthly_payment      DECIMAL(12,2) NOT NULL,
    dti                  DECIMAL(8,4) NOT NULL,            -- monthly_payment / monthly_income
    rf_probability       DECIMAL(6,4) NOT NULL,            -- Random Forest (decision model)
    lr_probability       DECIMAL(6,4) NOT NULL,
    dt_probability       DECIMAL(6,4) NOT NULL,
    risk_score           TINYINT UNSIGNED NOT NULL,        -- 0-100, higher = safer
    risk_level           ENUM('Low','Medium','High') NOT NULL,
    recommendation       VARCHAR(40) NOT NULL,
    top_factors          TEXT NULL,                        -- JSON list of key factors
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_application_applicant FOREIGN KEY (applicant_id)
        REFERENCES applicants(id) ON DELETE CASCADE,
    CONSTRAINT fk_application_user FOREIGN KEY (assessed_by)
        REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_created (created_at),
    INDEX idx_risk (risk_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
