import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
from sklearn.metrics import f1_score
from scipy.stats import ks_2samp   # drift detection

class AutoRetrainer:
    """
    Monitors model performance & data drift
    Automatically retrains when needed
    THIS IS YOUR FUTURE SCOPE MADE REAL!
    """
    def __init__(self, model, threshold_f1=0.90,
                 drift_threshold=0.05):
        self.model           = model
        self.threshold_f1    = threshold_f1
        self.drift_threshold = drift_threshold
        self.retrain_log     = []
        self.baseline_data   = None

    def set_baseline(self, X_train):
        """Store baseline data distribution"""
        # Store mean of each feature as baseline
        self.baseline_data = np.mean(X_train, axis=0)
        print("✅ Baseline set!")

    def detect_drift(self, X_new):
        """
        KS Test — checks if new data distribution
        has drifted from training data
        """
        if self.baseline_data is None:
            return False, 0.0

        drift_scores = []
        baseline_full = np.random.normal(
            self.baseline_data,
            0.1,
            (500, len(self.baseline_data))
        )

        for i in range(min(10, X_new.shape[1])):
            stat, p_value = ks_2samp(
                baseline_full[:, i],
                X_new[:, i]
            )
            drift_scores.append(p_value)

        avg_p_value = np.mean(drift_scores)
        drift_detected = avg_p_value < self.drift_threshold

        return drift_detected, avg_p_value

    def check_performance(self, X_test, y_test):
        """Check if model performance dropped"""
        y_pred    = self.model.rf.predict(X_test)
        current_f1 = f1_score(y_test, y_pred)

        print(f"Current F1: {current_f1:.4f} | "
              f"Threshold: {self.threshold_f1}")

        needs_retrain = current_f1 < self.threshold_f1
        return needs_retrain, current_f1

    def retrain(self, X_train, y_train,
                X_test, y_test, reason):
        """Retrain the model"""
        print(f"\n🔄 RETRAINING triggered: {reason}")

        self.model.train(X_train, y_train)
        metrics = self.model.evaluate(X_test, y_test)

        log_entry = {
            'timestamp'  : datetime.now().isoformat(),
            'reason'     : reason,
            'new_f1'     : metrics['f1'],
            'new_accuracy': metrics['accuracy']
        }
        self.retrain_log.append(log_entry)
        self._save_log()

        print(f"✅ Retrained! New F1: {metrics['f1']:.4f}")
        return metrics

    def auto_check_and_retrain(self, X_new, y_new,
                                X_train, y_train):
        """Main auto-check function"""
        results = {'retrained': False, 'reason': None}

        # Check 1 — Performance degradation
        needs_retrain, f1 = self.check_performance(
            X_new, y_new)

        if needs_retrain:
            metrics = self.retrain(
                X_train, y_train, X_new, y_new,
                reason=f"F1 dropped to {f1:.4f}"
            )
            results = {'retrained': True,
                       'reason'   : 'performance_drop',
                       'metrics'  : metrics}
            return results

        # Check 2 — Data drift
        drift, p_val = self.detect_drift(X_new)

        if drift:
            metrics = self.retrain(
                X_train, y_train, X_new, y_new,
                reason=f"Drift detected (p={p_val:.4f})"
            )
            results = {'retrained': True,
                       'reason'   : 'data_drift',
                       'metrics'  : metrics}
            return results

        print("✅ No retraining needed.")
        return results

    def _save_log(self):
        os.makedirs('logs', exist_ok=True)
        with open('logs/retrain_log.json', 'w') as f:
            json.dump(self.retrain_log, f, indent=2)

    def get_log(self):
        return self.retrain_log