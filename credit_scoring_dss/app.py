"""
Credit Scoring Decision Support System — application entry point.

Start it with:
    pip install -r requirements.txt
    python app.py
then open http://127.0.0.1:5000

MySQL stores user accounts, applicants and assessments, so it must be running.

Accounts
    First run:   a ready-made administrator exists: username "admin", password "admin".
                 Sign in and change both on the My account page.
    Officers:    click "Request an account" on the sign-in page; an administrator approves it.
    Admin:       the Users page adds accounts directly, approves requests, disables
                 accounts and resets passwords.
    Terminal:    flask --app app create-user   (optional)
"""
import json
import logging
import os
import re
import secrets
from functools import wraps
from urllib.parse import urlparse

import click
import pymysql
from flask import (Flask, abort, flash, redirect, render_template, request, session,
                   url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config
from model.tree_predictor import CreditModel
from routes.ml_routes import ml_bp

log = logging.getLogger('credit_dss')

# Pre-made administrator, created when the users table is empty. Change it after the first sign-in.
DEFAULT_ADMIN_USERNAME = 'admin'
DEFAULT_ADMIN_PASSWORD = 'admin'

USERNAME_RE = re.compile(r'^[A-Za-z0-9._-]{3,50}$')
MIN_PASSWORD = 8


class Database:
    """Small helper around PyMySQL that creates the database and tables on start-up."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.available = False
        self.error = None

    def connect(self, with_database=True):
        return pymysql.connect(
            host=self.cfg.MYSQL_HOST, port=self.cfg.MYSQL_PORT,
            user=self.cfg.MYSQL_USER, password=self.cfg.MYSQL_PASSWORD,
            database=self.cfg.MYSQL_DATABASE if with_database else None,
            charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor,
            autocommit=False, connect_timeout=3)

    def init(self):
        try:
            conn = self.connect(with_database=False)
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.cfg.MYSQL_DATABASE}` "
                            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                cur.execute(f"USE `{self.cfg.MYSQL_DATABASE}`")
                with open(self.cfg.SCHEMA_PATH) as fh:
                    sql = '\n'.join(line for line in fh if not line.strip().startswith('--'))
                for statement in sql.split(';'):
                    if statement.strip():
                        cur.execute(statement)
                cur.execute('SELECT COUNT(*) AS n FROM users')
                if cur.fetchone()['n'] == 0:
                    cur.execute('INSERT INTO users (username, full_name, password_hash, role, status) '
                                'VALUES (%s, %s, %s, %s, %s)',
                                (DEFAULT_ADMIN_USERNAME, 'Administrator',
                                 generate_password_hash(DEFAULT_ADMIN_PASSWORD), 'admin', 'active'))
                    print(f'\n  Created the default administrator: username "{DEFAULT_ADMIN_USERNAME}", '
                          f'password "{DEFAULT_ADMIN_PASSWORD}".\n  Sign in and change it on the My account page.\n')
            conn.commit()
            conn.close()
            self.available = True
            log.info('MySQL ready: database "%s"', self.cfg.MYSQL_DATABASE)
        except Exception as exc:          # MySQL not installed, not running, or wrong password
            self.available = False
            self.error = str(exc)
            log.warning('MySQL unavailable, sign-in is disabled: %s', exc)

    def run(self, sql, params=(), fetch=None):
        """Run one statement. fetch = 'one', 'all' or None (write)."""
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                result = cur.fetchone() if fetch == 'one' else cur.fetchall() if fetch == 'all' else None
            conn.commit()
            return result
        finally:
            conn.close()

    # ---------------------------------------------------------------- users
    def count_users(self):
        return self.run('SELECT COUNT(*) AS n FROM users', fetch='one')['n']

    def find_user(self, username):
        return self.run('SELECT id, username, full_name, password_hash, role, status '
                        'FROM users WHERE username = %s', (username,), fetch='one')

    def get_user(self, user_id):
        return self.run('SELECT id, username, full_name, password_hash, role, status '
                        'FROM users WHERE id = %s', (user_id,), fetch='one')

    def update_profile(self, user_id, username, full_name):
        self.run('UPDATE users SET username = %s, full_name = %s WHERE id = %s',
                 (username, full_name, user_id))

    def list_users(self):
        return self.run('SELECT id, username, full_name, role, status, last_login, created_at '
                        'FROM users ORDER BY FIELD(status, "pending", "active", "disabled"), full_name',
                        fetch='all')

    def count_pending(self):
        return self.run("SELECT COUNT(*) AS n FROM users WHERE status = 'pending'", fetch='one')['n']

    def create_user(self, username, full_name, password, role='officer', status='active'):
        self.run('INSERT INTO users (username, full_name, password_hash, role, status) '
                 'VALUES (%s, %s, %s, %s, %s)',
                 (username, full_name, generate_password_hash(password), role, status))

    def set_status(self, user_id, status):
        self.run('UPDATE users SET status = %s WHERE id = %s', (status, user_id))

    def set_password(self, user_id, password):
        self.run('UPDATE users SET password_hash = %s WHERE id = %s',
                 (generate_password_hash(password), user_id))

    def delete_pending(self, user_id):
        self.run("DELETE FROM users WHERE id = %s AND status = 'pending'", (user_id,))

    def record_login(self, user_id):
        self.run('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s', (user_id,))


def load_products(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


# ---------------------------------------------------------------- auth helpers
def current_user():
    return session.get('user')


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            return redirect(url_for('login', next=request.full_path.rstrip('?')))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if current_user()['role'] != 'admin':
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def csrf_token():
    if '_csrf' not in session:
        session['_csrf'] = secrets.token_urlsafe(24)
    return session['_csrf']


def check_csrf():
    sent = request.form.get('csrf_token', '')
    if not sent or not secrets.compare_digest(sent, session.get('_csrf', '')):
        abort(400, 'The form expired. Reload the page and try again.')


def safe_next(target):
    """Only redirect back to a page on this site after signing in."""
    if target and urlparse(target).netloc == '' and target.startswith('/') and not target.startswith('//'):
        return target
    return url_for('index')


def validate_account(form, need_role=False):
    """Check a new-account form. Returns (cleaned values, errors by field)."""
    values = {
        'username': form.get('username', '').strip(),
        'full_name': ' '.join(form.get('full_name', '').split()),
        'password': form.get('password', ''),
        'confirm': form.get('confirm', ''),
        'role': form.get('role', 'officer'),
    }
    errors = {}
    if not values['full_name']:
        errors['full_name'] = 'Enter the full name.'
    elif len(values['full_name']) > 120:
        errors['full_name'] = 'Use 120 characters or fewer.'
    if not USERNAME_RE.match(values['username']):
        errors['username'] = '3–50 characters: letters, numbers, dot, dash or underscore.'
    if len(values['password']) < MIN_PASSWORD:
        errors['password'] = f'Use at least {MIN_PASSWORD} characters.'
    elif values['password'] != values['confirm']:
        errors['confirm'] = 'The passwords do not match.'
    if need_role and values['role'] not in ('officer', 'admin'):
        errors['role'] = 'Choose a role.'
    return values, errors


def sign_in(user, db, password):
    session.clear()
    session.permanent = True
    session['user'] = {'id': user['id'], 'username': user['username'],
                       'full_name': user['full_name'], 'role': user['role'],
                       'default_password': password == DEFAULT_ADMIN_PASSWORD}
    db.record_login(user['id'])


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Shared resources, loaded once
    app.extensions['credit_model'] = CreditModel(app.config['MODEL_PATH'])
    app.extensions['catalog'] = load_products(app.config['PRODUCTS_PATH'])
    db = Database(config_class)
    db.init()
    app.extensions['db'] = db

    app.register_blueprint(ml_bp)

    tailwind_local = os.path.exists(os.path.join(app.static_folder, 'css', 'tailwind.css'))

    @app.context_processor
    def inject_globals():
        user = current_user()
        pending = 0
        if user and user['role'] == 'admin' and db.available:
            try:
                pending = db.count_pending()
            except Exception:
                pending = 0
        return {'active_page': request.endpoint, 'db_available': db.available,
                'tailwind_local': tailwind_local, 'user': user, 'pending_count': pending,
                'csrf_token': csrf_token, 'min_password': MIN_PASSWORD}

    @app.template_filter('peso')
    def peso(value):
        return f"₱{value:,.0f}"

    # ------------------------------------------------------------ sign in / out
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user():
            return redirect(url_for('index'))
        username = ''
        if request.method == 'POST':
            check_csrf()
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            if not db.available:
                flash('Sign-in is unavailable because MySQL is not connected. Start MySQL and restart the app.', 'error')
            elif not username or not password:
                flash('Enter your username and password.', 'error')
            else:
                try:
                    user = db.find_user(username)
                except Exception as exc:
                    log.warning('Login lookup failed: %s', exc)
                    flash('Could not reach the database. Try again.', 'error')
                else:
                    if user and check_password_hash(user['password_hash'], password):
                        if user['status'] == 'active':
                            sign_in(user, db, password)
                            return redirect(safe_next(request.args.get('next')))
                        flash('Your account is waiting for an administrator to approve it.'
                              if user['status'] == 'pending' else
                              'This account has been disabled. Contact the administrator.', 'error')
                    else:
                        flash('Incorrect username or password.', 'error')
        return render_template('login.html', username=username)

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user():
            return redirect(url_for('index'))
        values, errors = {}, {}
        if request.method == 'POST':
            check_csrf()
            values, errors = validate_account(request.form)
            if not db.available:
                flash('Accounts cannot be requested while MySQL is not connected.', 'error')
            elif not errors:
                try:
                    db.create_user(values['username'], values['full_name'], values['password'],
                                   'officer', 'pending')
                except pymysql.err.IntegrityError:
                    errors['username'] = 'That username is already taken. Choose another.'
                else:
                    flash(f'Request sent for "{values["username"]}". You can sign in once an '
                          'administrator approves it.', 'info')
                    return redirect(url_for('login'))
        return render_template('register.html', values=values, errors=errors)

    @app.post('/logout')
    def logout():
        check_csrf()
        session.clear()
        flash('You have signed out.', 'info')
        return redirect(url_for('login'))

    # ------------------------------------------------------------ my account
    @app.route('/account', methods=['GET', 'POST'])
    @login_required
    def account():
        me = current_user()
        profile_errors, password_errors = {}, {}
        if request.method == 'POST':
            check_csrf()
            if request.form.get('section') == 'profile':
                username = request.form.get('username', '').strip()
                full_name = ' '.join(request.form.get('full_name', '').split())
                if not full_name:
                    profile_errors['full_name'] = 'Enter your full name.'
                elif len(full_name) > 120:
                    profile_errors['full_name'] = 'Use 120 characters or fewer.'
                if not USERNAME_RE.match(username):
                    profile_errors['username'] = '3–50 characters: letters, numbers, dot, dash or underscore.'
                if not profile_errors:
                    try:
                        db.update_profile(me['id'], username, full_name)
                    except pymysql.err.IntegrityError:
                        profile_errors['username'] = 'That username is already taken.'
                    else:
                        session['user'] = {**me, 'username': username, 'full_name': full_name}
                        flash('Your name and username were updated. Use the new username next time you sign in.', 'info')
                        return redirect(url_for('account'))
            elif request.form.get('section') == 'password':
                current = request.form.get('current_password', '')
                new = request.form.get('new_password', '')
                confirm = request.form.get('confirm_password', '')
                stored = db.get_user(me['id'])
                if not stored or not check_password_hash(stored['password_hash'], current):
                    password_errors['current_password'] = 'This is not your current password.'
                if len(new) < MIN_PASSWORD:
                    password_errors['new_password'] = f'Use at least {MIN_PASSWORD} characters.'
                elif new == current:
                    password_errors['new_password'] = 'Choose a password different from the current one.'
                elif new != confirm:
                    password_errors['confirm_password'] = 'The passwords do not match.'
                if not password_errors:
                    db.set_password(me['id'], new)
                    session['user'] = {**me, 'default_password': False}
                    flash('Password changed. Use the new password next time you sign in.', 'info')
                    return redirect(url_for('account'))
        return render_template('account.html', profile_errors=profile_errors,
                               password_errors=password_errors,
                               form=request.form if request.method == 'POST' else {})

    # ------------------------------------------------------------ user management
    @app.route('/users', methods=['GET', 'POST'])
    @admin_required
    def users():
        values, errors = {}, {}
        if request.method == 'POST':
            check_csrf()
            values, errors = validate_account(request.form, need_role=True)
            if not errors:
                try:
                    db.create_user(values['username'], values['full_name'], values['password'],
                                   values['role'], 'active')
                except pymysql.err.IntegrityError:
                    errors['username'] = 'That username is already taken.'
                else:
                    flash(f'Created account "{values["username"]}". They can sign in now.', 'info')
                    return redirect(url_for('users'))
        return render_template('users.html', accounts=db.list_users(), values=values, errors=errors)

    @app.post('/users/<int:user_id>/<action>')
    @admin_required
    def user_action(user_id, action):
        check_csrf()
        if user_id == current_user()['id'] and action in ('disable', 'reject'):
            flash('You cannot disable your own account.', 'error')
        elif action in ('approve', 'enable'):
            db.set_status(user_id, 'active')
            flash('Account activated.', 'info')
        elif action == 'disable':
            db.set_status(user_id, 'disabled')
            flash('Account disabled. That person can no longer sign in.', 'info')
        elif action == 'reject':
            db.delete_pending(user_id)
            flash('Request rejected and removed.', 'info')
        elif action == 'password':
            password = request.form.get('password', '')
            if len(password) < MIN_PASSWORD:
                flash(f'New password must be at least {MIN_PASSWORD} characters.', 'error')
            else:
                db.set_password(user_id, password)
                flash('Password changed. Give the new password to the account owner.', 'info')
        else:
            abort(404)
        return redirect(url_for('users'))

    # ------------------------------------------------------------ page routes
    @app.route('/')
    @login_required
    def index():
        return render_template('index.html')

    @app.route('/predict')
    @login_required
    def predict():
        model = app.extensions['credit_model']
        return render_template(
            'predict.html', labels=model.labels, catalog=app.extensions['catalog'],
            selected_product=request.args.get('product', ''),
            defaults={'interest_rate_pa': app.config['DEFAULT_INTEREST_RATE'],
                      'loan_product': app.config['DEFAULT_LOAN_PRODUCT'],
                      'computation_mode': app.config['DEFAULT_COMPUTATION_MODE']},
            bands=model.meta['risk_bands'])

    @app.errorhandler(404)
    def not_found(_):
        return render_template('base.html', not_found=True), 404

    @app.errorhandler(403)
    def forbidden(_):
        return render_template('base.html', forbidden=True), 403

    # ------------------------------------------------------------ command line (optional)
    @app.cli.command('create-user')
    @click.option('--username', prompt=True)
    @click.option('--full-name', prompt='Full name')
    @click.option('--role', type=click.Choice(['officer', 'admin']), default='officer', prompt=True)
    @click.password_option()
    def create_user_command(username, full_name, role, password):
        """Add a loan officer or administrator account from the terminal."""
        if not db.available:
            raise click.ClickException(f'MySQL is not connected: {db.error}')
        try:
            db.create_user(username.strip(), full_name.strip(), password, role)
        except pymysql.err.IntegrityError:
            raise click.ClickException(f'The username "{username}" is already taken.')
        click.echo(f'Created {role} account "{username}".')

    return app


app = create_app()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    app.run(host='127.0.0.1', port=5000, debug=app.config['DEBUG'])
