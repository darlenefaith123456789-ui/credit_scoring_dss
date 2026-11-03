"""
API blueprint for the decision support system.

    POST /api/predict       score an applicant, save the assessment, return the result
    GET  /api/installment   estimate the monthly installment for a loan
    GET  /api/assessments   most recent saved assessments
"""
import json
import logging

from flask import Blueprint, current_app, jsonify, request, session

ml_bp = Blueprint('ml', __name__, url_prefix='/api')
log = logging.getLogger('credit_dss')

# Field -> (minimum, maximum) accepted by the API
NUMERIC_RULES = {
    'age': (18, 100),
    'monthly_income': (1, 10_000_000),
    'credit_amount': (1, 10_000_000),
    'term_months': (1, 120),
    'interest_rate_pa': (0, 100),
    'monthly_payment': (0, 10_000_000),
    'current_debt': (0, 100_000_000),
    'missed_payments_count': (0, 120),
    'late_payment_penalty': (0, 10_000_000),
}
CATEGORICAL_FIELDS = ['marital_status', 'employment_type', 'loan_product', 'computation_mode']

@ml_bp.before_request
def require_login():
    if not session.get('user'):
        return jsonify({'ok': False, 'error': 'Your session has ended. Sign in again.'}), 401


DTI_ALERT = 0.40          # installment above 40% of income is flagged
MISSED_PAYMENTS_ALERT = 3


# ------------------------------------------------------------------ calculations
def compute_installment(amount, rate_pa, term_months, computation_mode):
    """Estimated monthly installment.

    Flat rate / straight interest (codes 2, 3): interest on the full amount for the whole term.
    Diminishing (codes 0, 1): equal amortized payments on a declining balance.
    """
    amount, rate_pa, term_months = float(amount), float(rate_pa), int(term_months)
    if term_months <= 0:
        raise ValueError('term_months must be positive')
    if int(computation_mode) in (2, 3):
        interest = amount * rate_pa / 100 * term_months / 12
        return round((amount + interest) / term_months, 2)
    i = rate_pa / 100 / 12
    if i == 0:
        return round(amount / term_months, 2)
    return round(amount * i / (1 - (1 + i) ** -term_months), 2)


def compute_dti(monthly_payment, monthly_income):
    """Debt-to-Income ratio = monthly installment / monthly income (0.25 = 25% of income)."""
    monthly_income = float(monthly_income)
    return float(monthly_payment) / monthly_income if monthly_income > 0 else None


# ------------------------------------------------------------------ validation
def parse_application(data, model):
    record, errors = {}, {}
    for field, (lo, hi) in NUMERIC_RULES.items():
        raw = data.get(field)
        if field == 'monthly_payment' and raw in (None, ''):
            continue                      # computed below when left blank
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[field] = 'Enter a number.'
            continue
        if not lo <= value <= hi:
            errors[field] = f'Must be between {lo:,} and {hi:,}.'
        record[field] = value

    for field in CATEGORICAL_FIELDS:
        allowed = model.prep['cats'][model.prep['cat_cols'].index(field)]
        try:
            value = int(data.get(field))
        except (TypeError, ValueError):
            errors[field] = 'Choose an option.'
            continue
        if value not in allowed:
            errors[field] = 'Choose an option from the list.'
        record[field] = value

    if not errors and 'monthly_payment' not in record:
        record['monthly_payment'] = compute_installment(
            record['credit_amount'], record['interest_rate_pa'],
            record['term_months'], record['computation_mode'])
    return record, errors


# ------------------------------------------------------------------ persistence
def save_assessment(data, record, result):
    db = current_app.extensions['db']
    if not db.available:
        return None
    try:
        conn = db.connect()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO applicants (applicant_name, age, marital_status, employment_type,
                       monthly_income, current_debt, missed_payments_count, late_payment_penalty)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                ((data.get('applicant_name') or '').strip()[:120] or None,
                 int(record['age']), record['marital_status'], record['employment_type'],
                 record['monthly_income'], record['current_debt'],
                 int(record['missed_payments_count']), record['late_payment_penalty']))
            applicant_id = cur.lastrowid
            p = result['probabilities']
            cur.execute(
                """INSERT INTO loan_applications (applicant_id, assessed_by, product_id, product_name,
                       loan_product, credit_amount, term_months, interest_rate_pa,
                       computation_mode, monthly_payment, dti, rf_probability, lr_probability,
                       dt_probability, risk_score, risk_level, recommendation, top_factors)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (applicant_id, session['user']['id'], data.get('product_id') or None, data.get('product_name') or None,
                 record['loan_product'], record['credit_amount'], int(record['term_months']),
                 record['interest_rate_pa'], record['computation_mode'], record['monthly_payment'],
                 round(record['dti'], 4), round(p['random_forest'], 4),
                 round(p['logistic_regression'], 4), round(p['decision_tree'], 4),
                 result['risk_score'], result['risk_level'], result['recommendation'],
                 json.dumps(result['factors'])))
            application_id = cur.lastrowid
        conn.commit()
        conn.close()
        return application_id
    except Exception as exc:
        log.warning('Could not save assessment: %s', exc)
        return None


# ------------------------------------------------------------------ endpoints
@ml_bp.post('/predict')
def predict():
    model = current_app.extensions['credit_model']
    data = request.get_json(silent=True) or request.form.to_dict()
    record, errors = parse_application(data, model)
    if errors:
        return jsonify({'ok': False, 'errors': errors}), 400

    record['dti'] = compute_dti(record['monthly_payment'], record['monthly_income'])
    result = model.predict(record)

    alerts = []
    if record['dti'] > DTI_ALERT:
        alerts.append(f"Installment takes {record['dti']:.0%} of monthly income "
                      f"(above the {DTI_ALERT:.0%} guideline).")
    if record['missed_payments_count'] >= MISSED_PAYMENTS_ALERT:
        alerts.append(f"{int(record['missed_payments_count'])} missed payments on record.")

    application_id = save_assessment(data, record, result)
    return jsonify({
        'ok': True,
        'application_id': application_id,
        'saved': application_id is not None,
        'monthly_payment': record['monthly_payment'],
        'dti': record['dti'],
        'alerts': alerts,
        **result,
    })


@ml_bp.get('/installment')
def installment():
    try:
        value = compute_installment(request.args['amount'], request.args['rate'],
                                    request.args['term'], request.args['mode'])
    except (KeyError, ValueError, ZeroDivisionError):
        return jsonify({'ok': False, 'error': 'Provide amount, rate, term and mode.'}), 400
    return jsonify({'ok': True, 'monthly_payment': value})


@ml_bp.get('/assessments')
def assessments():
    db = current_app.extensions['db']
    if not db.available:
        return jsonify({'ok': True, 'saved': False, 'items': []})
    limit = max(1, min(int(request.args.get('limit', 10)), 100))
    try:
        conn = db.connect()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT la.id, a.applicant_name, la.product_name, la.credit_amount,
                          la.term_months, la.dti, la.rf_probability, la.risk_level,
                          la.recommendation, la.created_at, u.full_name AS officer
                   FROM loan_applications la
                   JOIN applicants a ON a.id = la.applicant_id
                   LEFT JOIN users u ON u.id = la.assessed_by
                   ORDER BY la.id DESC LIMIT %s""", (limit,))
            rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        log.warning('Could not read assessments: %s', exc)
        return jsonify({'ok': False, 'error': 'Could not read saved assessments.'}), 500
    for r in rows:
        for k in ('credit_amount', 'dti', 'rf_probability'):
            r[k] = float(r[k])
        r['created_at'] = r['created_at'].strftime('%Y-%m-%d %H:%M')
    return jsonify({'ok': True, 'saved': True, 'items': rows})
