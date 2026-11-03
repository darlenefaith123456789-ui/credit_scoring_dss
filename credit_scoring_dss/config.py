"""
Database and environment configuration.

Values come from environment variables so passwords stay out of the code.
On Windows (Command Prompt):   set MYSQL_PASSWORD=yourpassword
On Windows (PowerShell):       $env:MYSQL_PASSWORD="yourpassword"
On macOS / Linux:              export MYSQL_PASSWORD=yourpassword
"""
import os
import secrets
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # Signs the login cookie. Set SECRET_KEY to keep users signed in across restarts.
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)   # one working day
    DEBUG = os.environ.get('FLASK_DEBUG', '1') == '1'

    # MySQL (XAMPP / WAMP defaults: user "root", empty password, port 3306)
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.environ.get('MYSQL_PORT', '3306'))
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'credit_scoring_dss')

    # Files
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'pure_json_tree.json')
    PRODUCTS_PATH = os.path.join(BASE_DIR, 'products.json')
    SCHEMA_PATH = os.path.join(BASE_DIR, 'schema.sql')

    # Defaults used to pre-fill the application form
    DEFAULT_INTEREST_RATE = 24.0     # % per annum (median of the dataset)
    DEFAULT_LOAN_PRODUCT = 5         # "Regular loan"
    DEFAULT_COMPUTATION_MODE = 2     # "Flat rate"
