import shap
import numpy as np
import matplotlib.pyplot as plt
import pickle

class ExplainableIDS:
    """
    SHAP Explainability — answers WHY model flagged traffic
    This is your biggest differentiator in the project!
    """
    def __init__(self, model, feature_names):
        self.model         = model
        self.feature_names = feature_names
        self.explainer     = None

    def setup_explainer(self, X_train_sample):
        print("Setting up SHAP explainer...")
        # Use small sample for low RAM
        sample = X_train_sample[:500]
        self.explainer = shap.TreeExplainer(self.model)
        print("✅ Explainer ready!")

    def explain_prediction(self, X_input):
        """Explain a single prediction"""
        shap_values = self.explainer.shap_values(X_input)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = attack
        return shap_values

    def plot_summary(self, X_test_sample, save=True):
        """Global feature importance via SHAP"""
        print("Generating SHAP summary plot...")
        shap_values = self.explainer.shap_values(
            X_test_sample[:200])

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_test_sample[:200],
            feature_names = self.feature_names,
            show          = False
        )
        if save:
            plt.savefig('reports/shap_summary.png',
                        bbox_inches='tight')
            print("Saved: reports/shap_summary.png")
        plt.show()

    def plot_single_explanation(self, X_input,
                                 pred_label, save=True):
        """Explain ONE prediction — shows in dashboard"""
        shap_values = self.explain_prediction(X_input)

        plt.figure(figsize=(12, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values        = shap_values[0],
                base_values   = self.explainer.expected_value,
                data          = X_input[0],
                feature_names = self.feature_names
            ),
            show = False
        )
        title = "⚠️ ATTACK" if pred_label == 1 else "✅ NORMAL"
        plt.title(f"Prediction Explanation: {title}")
        if save:
            plt.savefig('reports/shap_single.png',
                        bbox_inches='tight')
        plt.show()

    def get_top_features(self, X_input, top_n=5):
        """Return top N reasons for prediction"""
        shap_vals = self.explain_prediction(X_input)[0]
        indices   = np.argsort(np.abs(shap_vals))[::-1][:top_n]

        reasons = []
        for i in indices:
            reasons.append({
                'feature'   : self.feature_names[i],
                'value'     : float(X_input[0][i]),
                'impact'    : float(shap_vals[i]),
                'direction' : '⬆️ Increases Attack Risk'
                               if shap_vals[i] > 0
                               else '⬇️ Decreases Attack Risk'
            })
        return reasons