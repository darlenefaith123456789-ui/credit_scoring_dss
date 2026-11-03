# Credit Scoring Decision Support System

**Data Mining–Based Credit Scoring Decision Support System for Installment-Based Financing**
Computer Science Department, Quirino State University · AY 2025–2026

A Flask web application that helps loan officers assess installment-financing applicants.
It estimates each applicant's chance of default with a Random Forest model, assigns a risk
level, recommends **Approve / Refer for review / Reject**, and shows the factors behind the result.

---

## 1. Requirements

- Python 3.10 or newer
- MySQL 8 or MariaDB (XAMPP or WAMP is fine). Required: it stores the user accounts,
  applicants and assessments.
- Internet access for Tailwind styling, or the offline build in section 6

## 2. Setup

```bash
cd credit_scoring_dss
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Start MySQL (in XAMPP: **Start** beside MySQL). The default settings in `config.py` match XAMPP
(user `root`, no password, port 3306). If your MySQL has a password, set it before starting:

```bash
set MYSQL_PASSWORD=yourpassword          # Windows Command Prompt
$env:MYSQL_PASSWORD="yourpassword"       # Windows PowerShell
export MYSQL_PASSWORD=yourpassword       # macOS / Linux
```

The database `credit_scoring_dss` and its tables are created automatically on start-up.

## 3. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** and sign in.

| Page | What it does |
|---|---|
| Sign in `/login` | Username and password for loan officers and administrators |
| Request an account `/register` | New loan officers ask for an account; an administrator approves it |
| My account `/account` | Change your own name, username and password |
| Users `/users` | Administrators only: add, approve, disable accounts and reset passwords |
| Home `/` | System overview, workflow, model comparison, strongest predictors |
| Assess applicant `/predict` | Application form and recommendation dashboard, plus recent saved assessments with the officer who made each one |

Every page and API endpoint requires signing in. Sessions last 8 hours.

## 3a. User accounts

Everything is done in the browser.

**Default administrator.** The first time the app starts, it creates an administrator account:

| Username | Password |
|---|---|
| `admin` | `admin` |

Sign in with it, then open **My account** (click your name in the top menu) to change the password, and
the username and name if you like. Until the password is changed, a red bar on every page reminds you.
Every user can change their own name, username and password on **My account**.

**New loan officers** click **Request an account** on the sign-in page and fill in their name, username and
password. The request waits for approval; until then that person sees "waiting for an administrator to approve it".

**Administrators** use the **Users** page (in the top menu; a red badge shows waiting requests) to:
- approve or reject account requests
- add an account directly (it works immediately)
- disable or re-enable an account
- reset a forgotten password

Optional terminal command, for example on a server: `flask --app app create-user`.

**Stay signed in after a restart.** The login cookie is signed with `SECRET_KEY`. If it is not set, a new
key is made on every start and everyone has to sign in again. Set a long random value to avoid that:
`set SECRET_KEY=any-long-random-text`.

Passwords are stored as salted hashes (Werkzeug `scrypt`/`pbkdf2`), never as plain text. The sign-in and
sign-out forms carry a CSRF token.

**Upgrading from an earlier version:** if you already created the `credit_scoring_dss` database,
drop it once (`DROP DATABASE credit_scoring_dss;` in phpMyAdmin) so the `users` table (with its `status`
column) and the `assessed_by` column are created.

## 4. How the scoring works

1. **Input:** applicant profile, income and debt, the loan, and repayment record.
2. **Installment:** estimated from amount, rate, term and computation method
   (flat rate / straight interest: add-on interest; diminishing: amortized). The officer can type over it.
3. **Debt-to-Income (DTI):** `monthly installment ÷ monthly income`. It is an engineered feature
   the model uses, and an alert appears when it is above 40%.
4. **Prediction:** Random Forest (300 trees) gives the probability of a default record.
5. **Risk level:** Low under 30% → Approve · Medium 30–50% → Refer for review · High 50%+ → Reject.
   The bands are set in `model/train_models.py` (`risk_bands`).
6. **Key factors:** each tree's decision path is split by input field, which shows how much each
   field pushed the score up or down from the model's 50% starting point.
   (The starting point is 50% because training used balanced class weights.)

The Logistic Regression and Decision Tree probabilities are shown beside the Random Forest result for comparison.

### Target used for training
`previous_defaults` from the cleaned dataset: yes = 1, no = 0. The 360 "unknown" records are excluded.
The model therefore predicts whether an applicant has a **recorded previous default**.

### Current results (670 test records, 50% cut-off)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (used) | 0.861 | 0.587 | 0.740 | 0.654 | 0.871 |
| Logistic Regression | 0.822 | 0.500 | 0.765 | 0.605 | 0.869 |
| Decision Tree | 0.785 | 0.434 | 0.689 | 0.533 | 0.796 |

At the 30% "Refer for review" cut-off, Random Forest catches 86.6% of applicants with a default record.

## 5. Retraining the models

```bash
python model/train_models.py
```

This reads `utils/dataset.csv`, repeats the stratified 80:20 split (random_state = 42), trains all
three models, prints the comparison, and writes:

- `model/pure_json_tree.json`: every fitted model as plain JSON (what the web app loads)
- `model/model_comparison.csv`: the metrics table

The web app does **not** need scikit-learn. `model/tree_predictor.py` runs the exported trees in pure
Python and reproduces scikit-learn's probabilities exactly, so a scikit-learn upgrade cannot break the
running system.

To train on a different file: `python model/train_models.py --data path/to/file.csv`.
The file must have the same columns as `utils/dataset.csv` (the cleaning notebook's output).

## 6. Running without internet (for the defense)

Styling uses the Tailwind CDN by default. To build it locally once (needs Node.js and internet that one time):

```bash
npx tailwindcss@3.4.17 -i static/css/tailwind.input.css -o static/css/tailwind.css --minify
```

When `static/css/tailwind.css` exists, the app uses it automatically and no longer needs the CDN.
Fonts fall back to system fonts when offline.

## 7. API

| Method | URL | Purpose |
|---|---|---|
| POST | `/api/predict` | JSON body with the form fields → probability, risk level, recommendation, DTI, alerts, key factors |
| GET | `/api/installment?amount=&rate=&term=&mode=` | Estimated monthly installment |
| GET | `/api/assessments?limit=10` | Most recent saved assessments |

All API calls require a signed-in session; otherwise they return `401`.

## 8. Category codes

Categorical fields are sent as codes (the cleaning notebook's alphabetical mapping):

- **marital_status:** 0 Divorced, 1 Married, 2 Single, 3 Widowed
- **computation_mode:** 0 Diminishing, 1 Diminishing with increasing principal, 2 Flat rate, 3 Straight interest
- **loan_product:** 0 Agri-grain, 1 Commercial, 2 Microfinance (agri), 3 Microfinance (non-agri), 4 Productive (regular), 5 Regular
- **employment_type:** 17 occupations. The full list is in `LABELS` in `model/train_models.py`.

## 9. Notes

- `utils/dataset.csv` is the cleaned thesis dataset (3,710 records), the output of `clean_thesis_datasets_v2_colab.ipynb`.
- `products.json` is a **sample** motorcycle list used by the form's Motorcycle picker (which fills in the credit amount).
  Replace the names, prices and financing terms with the partner dealer's price list.
- The installments recorded in the historical data do not follow one standard formula, so the system's
  estimate may differ from past records. Officers can enter the actual installment by hand.
- This is a decision support tool. The loan officer makes the final decision.
