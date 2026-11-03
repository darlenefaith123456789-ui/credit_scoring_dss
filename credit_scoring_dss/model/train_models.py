"""
Model trainer for the Credit Scoring Decision Support System.

Trains and compares Logistic Regression, Decision Tree and Random Forest on the
cleaned thesis dataset, then exports every fitted model to a portable JSON file
(model/pure_json_tree.json). The web app reads only that JSON file, so it runs
without scikit-learn and never breaks because of a scikit-learn version change.

Run from the project root:
    python model/train_models.py
    python model/train_models.py --data utils/dataset.csv --out model/pure_json_tree.json

Target definition
    previous_defaults: 0 = no, 1 = unknown, 2 = yes   (cleaning-notebook codes)
    credit_risk_target = 1 if previous_defaults == 2, 0 if == 0; unknown rows are dropped.
    The model therefore predicts a RECORDED PREVIOUS DEFAULT, which is used as the
    credit-risk label in this study.

Engineered feature
    dti = monthly_payment / monthly_income   (Debt-to-Income ratio, see routes/ml_routes.py)
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_STATE = 42

NUMERIC_COLS = ['age', 'monthly_income', 'credit_amount', 'term_months', 'interest_rate_pa',
                'monthly_payment', 'current_debt', 'missed_payments_count',
                'late_payment_penalty', 'dti']
CATEGORICAL_COLS = ['marital_status', 'employment_type', 'loan_product', 'computation_mode']

# Code -> label mapping produced by the cleaning notebook (alphabetical order of the
# cleaned text values). Verified against thesis_datasets.csv on all 3,710 records.
LABELS = {
    'marital_status': ['Divorced', 'Married', 'Single', 'Widowed'],
    'employment_type': ['Agrarian beneficiary', 'Agri-vendor', 'Agricultural worker',
                        'Banana planter', 'Civil servant', 'Corn farmer', 'Driver', 'Employed',
                        'Farm worker', 'Farmer', 'Livestock farmer', 'Poultry raiser',
                        'Rice farmer', 'Self-employed', 'Small merchant / vendor', 'Teacher',
                        'Vegetable farmer'],
    'loan_product': ['Agri-grain loan', 'Commercial loan', 'Microfinance loan (agri)',
                     'Microfinance loan (non-agri)', 'Productive loan (regular)', 'Regular loan'],
    'computation_mode': ['Diminishing', 'Diminishing with increasing principal', 'Flat rate',
                         'Straight interest'],
}

FIELD_NAMES = {
    'age': 'Age', 'monthly_income': 'Monthly income', 'credit_amount': 'Credit amount',
    'term_months': 'Term', 'interest_rate_pa': 'Interest rate',
    'monthly_payment': 'Monthly installment', 'current_debt': 'Current debt',
    'missed_payments_count': 'Missed payments', 'late_payment_penalty': 'Late payment penalty',
    'dti': 'Debt-to-income ratio', 'marital_status': 'Marital status',
    'employment_type': 'Occupation', 'loan_product': 'Loan product',
    'computation_mode': 'Computation mode',
}


def load_dataset(path):
    df = pd.read_csv(path)
    known = df[df['previous_defaults'].isin([0, 2])].copy()
    known['credit_risk_target'] = (known['previous_defaults'] == 2).astype(int)
    # Feature engineering: Debt-to-Income ratio (same formula as the API)
    known['dti'] = known['monthly_payment'] / known['monthly_income'].where(known['monthly_income'] > 0)
    X = known[NUMERIC_COLS + CATEGORICAL_COLS]
    y = known['credit_risk_target']
    print(f'Loaded {len(df)} rows; {len(known)} with a known target '
          f'({len(df) - len(known)} unknown excluded).')
    print('Class counts (0 = no default record, 1 = default record):',
          y.value_counts().sort_index().to_dict())
    return X, y


def build_preprocessor():
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('numeric', numeric, NUMERIC_COLS),
                              ('categorical', categorical, CATEGORICAL_COLS)])


def build_models():
    return {
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced',
                                                  random_state=RANDOM_STATE),
        'decision_tree': DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=5,
                                                random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                                min_samples_leaf=2, n_jobs=-1,
                                                random_state=RANDOM_STATE),
    }


def export_tree(tree):
    """Convert a fitted sklearn tree into plain lists (left, right, feature, threshold, P(default))."""
    t = tree.tree_
    value = t.value[:, 0, :]
    value = value / value.sum(axis=1, keepdims=True)
    return {
        'l': t.children_left.tolist(),
        'r': t.children_right.tolist(),
        'f': t.feature.tolist(),
        'th': [float(v) for v in t.threshold],
        'p': [round(float(v), 8) for v in value[:, 1]],
    }


def export_preprocessing(pre):
    num = pre.named_transformers_['numeric']
    cat = pre.named_transformers_['categorical']
    return {
        'num_cols': NUMERIC_COLS,
        'num_fill': num.named_steps['imputer'].statistics_.tolist(),
        'mean': num.named_steps['scaler'].mean_.tolist(),
        'scale': num.named_steps['scaler'].scale_.tolist(),
        'cat_cols': CATEGORICAL_COLS,
        'cat_fill': [int(v) for v in cat.named_steps['imputer'].statistics_],
        'cats': [[int(v) for v in c] for c in cat.named_steps['onehot'].categories_],
    }


def evaluate(name, y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'model': name,
        'accuracy': round(accuracy_score(y_true, pred), 4),
        'precision': round(precision_score(y_true, pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_true, proba), 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'utils', 'dataset.csv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'model', 'pure_json_tree.json'))
    args = ap.parse_args()

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    print(f'Train: {len(X_train)} rows | Test: {len(X_test)} rows')

    fitted, metrics, probas = {}, [], {}
    for name, estimator in build_models().items():
        pipe = Pipeline([('preprocess', build_preprocessor()), ('model', estimator)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        fitted[name], probas[name] = pipe, proba
        metrics.append(evaluate(name, y_test, proba))

    print('\nTest-set results (50% cut-off):')
    print(pd.DataFrame(metrics).drop(columns='confusion_matrix').to_string(index=False))
    rf_proba = probas['random_forest']
    print('\nRandom Forest recall at the 30% "Review" cut-off: '
          f'{recall_score(y_test, (rf_proba >= 0.30).astype(int)):.3f}')

    rf = fitted['random_forest']
    pre = rf.named_steps['preprocess']
    feature_names = NUMERIC_COLS + [f'{c}={v}' for c, cats in
                                    zip(CATEGORICAL_COLS, export_preprocessing(pre)['cats'])
                                    for v in cats]
    importance = sorted(zip(feature_names, rf.named_steps['model'].feature_importances_),
                        key=lambda t: -t[1])
    print('\nTop Random Forest features:')
    for f, v in importance[:8]:
        print(f'  {f:28s} {v:.3f}')

    lr = fitted['logistic_regression'].named_steps['model']
    export = {
        'meta': {
            'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'dataset': os.path.basename(args.data),
            'train_rows': int(len(X_train)),
            'test_rows': int(len(X_test)),
            'target': 'Recorded previous default (previous_defaults: 2 = yes -> 1, 0 = no -> 0)',
            'primary_model': 'random_forest',
            'risk_bands': {'low_below': 0.30, 'high_from': 0.50},
        },
        'labels': LABELS,
        'field_names': FIELD_NAMES,
        'metrics': metrics,
        'feature_importance': [{'feature': f, 'importance': round(float(v), 4)}
                               for f, v in importance],
        # All three pipelines share the same preprocessing settings (fitted on the same data)
        'prep': export_preprocessing(pre),
        'models': {
            'random_forest': {'type': 'forest',
                              'trees': [export_tree(t) for t in rf.named_steps['model'].estimators_]},
            'decision_tree': {'type': 'tree',
                              'tree': export_tree(fitted['decision_tree'].named_steps['model'])},
            'logistic_regression': {'type': 'linear', 'coef': lr.coef_[0].tolist(),
                                    'intercept': float(lr.intercept_[0])},
        },
    }
    with open(args.out, 'w') as fh:
        json.dump(export, fh, separators=(',', ':'))
    pd.DataFrame(metrics).drop(columns='confusion_matrix').to_csv(
        os.path.join(os.path.dirname(args.out), 'model_comparison.csv'), index=False)
    print(f'\nSaved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) and model_comparison.csv')

    # Self-check: the portable JSON must reproduce sklearn's probabilities
    import sys
    sys.path.insert(0, ROOT)
    from model.tree_predictor import CreditModel
    portable = CreditModel(args.out)
    sample = X_test.head(50).to_dict('records')
    diff = max(abs(portable.predict(r)['probabilities']['random_forest'] - p)
               for r, p in zip(sample, rf_proba[:50]))
    print(f'Portable model check (max difference on 50 test rows): {diff:.2e}')


if __name__ == '__main__':
    main()
