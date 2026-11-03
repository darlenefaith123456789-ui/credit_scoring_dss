"""
Standalone executor for the exported models in pure_json_tree.json.

Pure Python: needs no scikit-learn, numpy or pandas, so the web app keeps working
whatever library versions are installed. It reproduces sklearn's predict_proba
exactly (same preprocessing, same tree splits).

Usage:
    from model.tree_predictor import CreditModel
    model = CreditModel('model/pure_json_tree.json')
    result = model.predict({'age': 35, 'marital_status': 1, ...})
"""
import json
import math
import os
from array import array

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pure_json_tree.json')


class CreditModel:
    def __init__(self, path=DEFAULT_PATH):
        with open(path) as fh:
            data = json.load(fh)
        self.meta = data['meta']
        self.labels = data['labels']
        self.field_names = data['field_names']
        self.metrics = data['metrics']
        self.feature_importance = data['feature_importance']
        self.prep = data['prep']
        self.models = data['models']
        self.input_fields = self.prep['num_cols'] + self.prep['cat_cols']

        # For each column of the model's input vector, remember which form field it came from
        self._owner = list(self.prep['num_cols'])
        for col, cats in zip(self.prep['cat_cols'], self.prep['cats']):
            self._owner.extend([col] * len(cats))

        trees = self.models['random_forest']['trees']
        self.base_rate = sum(t['p'][0] for t in trees) / len(trees)
        bands = self.meta['risk_bands']
        self.low_below, self.high_from = bands['low_below'], bands['high_from']

    # ---------------------------------------------------------------- preprocessing
    def vectorize(self, record):
        """Scale numeric fields and one-hot encode categorical fields, as in training."""
        p, z = self.prep, []
        for i, col in enumerate(p['num_cols']):
            v = record.get(col)
            v = p['num_fill'][i] if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
            z.append((v - p['mean'][i]) / p['scale'][i])
        for i, col in enumerate(p['cat_cols']):
            v = record.get(col)
            v = p['cat_fill'][i] if v is None else int(v)
            z.extend(1.0 if v == c else 0.0 for c in p['cats'][i])
        return z

    @staticmethod
    def _float32(z):
        # sklearn trees compare float32 inputs; round the same way so splits match exactly
        return array('f', z).tolist()

    # ---------------------------------------------------------------- tree walking
    def _walk(self, tree, z, contrib=None):
        n = 0
        left, right, feat, th, prob = tree['l'], tree['r'], tree['f'], tree['th'], tree['p']
        while left[n] != -1:
            nxt = left[n] if z[feat[n]] <= th[n] else right[n]
            if contrib is not None:
                contrib[self._owner[feat[n]]] += prob[nxt] - prob[n]
            n = nxt
        return prob[n]

    def _forest(self, z):
        """Random Forest probability plus per-field contributions (path decomposition).

        base_rate + sum(contributions) == probability, so each contribution shows how
        far that field moved this applicant away from the average applicant.
        """
        trees = self.models['random_forest']['trees']
        contrib = {f: 0.0 for f in self.input_fields}
        total = sum(self._walk(t, z, contrib) for t in trees)
        n = len(trees)
        return total / n, {f: v / n for f, v in contrib.items()}

    def _linear(self, z):
        m = self.models['logistic_regression']
        s = m['intercept'] + sum(a * b for a, b in zip(z, m['coef']))
        return 1.0 / (1.0 + math.exp(-s))

    # ---------------------------------------------------------------- public API
    def risk_level(self, probability):
        if probability < self.low_below:
            return 'Low', 'Approve'
        if probability < self.high_from:
            return 'Medium', 'Refer for manual review'
        return 'High', 'Reject'

    def predict(self, record, top_n=5):
        z = self.vectorize(record)
        z32 = self._float32(z)
        rf_p, contrib = self._forest(z32)
        level, recommendation = self.risk_level(rf_p)
        factors = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:top_n]
        return {
            'probability': rf_p,
            'risk_score': round((1 - rf_p) * 100),   # 100 = safest, 0 = riskiest
            'risk_level': level,
            'recommendation': recommendation,
            'base_rate': self.base_rate,
            'probabilities': {
                'random_forest': rf_p,
                'logistic_regression': self._linear(z),
                'decision_tree': self._walk(self.models['decision_tree']['tree'], z32),
            },
            'factors': [{'field': f, 'name': self.field_names.get(f, f),
                         'effect': round(v, 4),
                         'direction': 'raises risk' if v > 0 else 'lowers risk'}
                        for f, v in factors],
        }


if __name__ == '__main__':
    model = CreditModel()
    demo = {'age': 42, 'marital_status': 1, 'employment_type': 12, 'monthly_income': 18000,
            'loan_product': 2, 'credit_amount': 50000, 'term_months': 18, 'interest_rate_pa': 24,
            'computation_mode': 0, 'monthly_payment': 3400, 'current_debt': 15000,
            'missed_payments_count': 2, 'late_payment_penalty': 300, 'dti': 3400 / 18000}
    out = model.predict(demo)
    print(f"Default probability: {out['probability']:.1%} -> {out['risk_level']} ({out['recommendation']})")
    for f in out['factors']:
        print(f"  {f['name']:24s} {f['effect'] * 100:+.1f} pts")
