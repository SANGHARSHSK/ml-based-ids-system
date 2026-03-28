import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
    precision_score, recall_score,
    roc_auc_score, confusion_matrix,
    classification_report)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense,
    Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Limit GPU/CPU memory for low-end PC
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


class RandomForestIDS:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators = 50,    # light for low RAM
            max_depth    = 8,
            random_state = 42,
            n_jobs       = 1
        )

    def train(self, X_train, y_train):
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        print("✅ Random Forest trained!")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:,1]
        return self._metrics(y_test, y_pred, y_prob)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path='models/rf_model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ RF Model saved to {path}")

    def load(self, path='models/rf_model.pkl'):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def _metrics(self, y_test, y_pred, y_prob):
        return {
            'accuracy'  : accuracy_score(y_test, y_pred),
            'precision' : precision_score(y_test, y_pred),
            'recall'    : recall_score(y_test, y_pred),
            'f1'        : f1_score(y_test, y_pred),
            'roc_auc'   : roc_auc_score(y_test, y_prob),
            'conf_matrix': confusion_matrix(y_test, y_pred)
        }


class LSTMIDS:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model     = self._build_model()

    def _build_model(self):
        # Light LSTM — safe for 4-8GB RAM
        model = Sequential([
            # Reshape for LSTM input
            tf.keras.layers.Reshape(
                (1, self.input_dim),
                input_shape=(self.input_dim,)),

            LSTM(32, return_sequences=True),
            Dropout(0.2),

            LSTM(16),
            Dropout(0.2),

            BatchNormalization(),

            Dense(16, activation='relu'),
            Dense(1,  activation='sigmoid')
        ])

        model.compile(
            optimizer = 'adam',
            loss      = 'binary_crossentropy',
            metrics   = ['accuracy']
        )
        return model

    def train(self, X_train, y_train):
        print("Training LSTM...")
        early_stop = EarlyStopping(
            monitor  = 'val_loss',
            patience = 3,
            restore_best_weights = True
        )
        self.history = self.model.fit(
            X_train, y_train,
            epochs          = 10,
            batch_size      = 256,
            validation_split= 0.1,
            callbacks       = [early_stop],
            verbose         = 1
        )
        print("✅ LSTM trained!")

    def evaluate(self, X_test, y_test):
        y_prob = self.model.predict(X_test).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        return {
            'accuracy'  : accuracy_score(y_test, y_pred),
            'precision' : precision_score(y_test, y_pred),
            'recall'    : recall_score(y_test, y_pred),
            'f1'        : f1_score(y_test, y_pred),
            'roc_auc'   : roc_auc_score(y_test, y_prob),
            'conf_matrix': confusion_matrix(y_test, y_pred)
        }

    def predict_proba(self, X):
        return self.model.predict(X).flatten()

    def save(self, path='models/lstm_model'):
        self.model.save(path)
        print(f"✅ LSTM saved to {path}")

    def load(self, path='models/lstm_model'):
        self.model = tf.keras.models.load_model(path)


class HybridEnsembleIDS:
    """
    Combines RF + LSTM predictions
    Final prediction = weighted average of both
    This is the MOST ADVANCED part of your project!
    """
    def __init__(self, rf_weight=0.6, lstm_weight=0.4):
        self.rf         = RandomForestIDS()
        self.rf_weight  = rf_weight
        self.lstm_weight= lstm_weight
        self.lstm       = None

    def train(self, X_train, y_train):
        # Train RF
        self.rf.train(X_train, y_train)

        # Train LSTM
        self.lstm = LSTMIDS(X_train.shape[1])
        self.lstm.train(X_train, y_train)

    def predict(self, X):
        rf_prob   = self.rf.predict_proba(X)[:,1]
        lstm_prob = self.lstm.predict_proba(X)

        # Weighted ensemble
        final_prob = (self.rf_weight   * rf_prob +
                      self.lstm_weight * lstm_prob)
        return (final_prob > 0.5).astype(int), final_prob

    def evaluate(self, X_test, y_test):
        y_pred, y_prob = self.predict(X_test)
        return {
            'accuracy'  : accuracy_score(y_test, y_pred),
            'precision' : precision_score(y_test, y_pred),
            'recall'    : recall_score(y_test, y_pred),
            'f1'        : f1_score(y_test, y_pred),
            'roc_auc'   : roc_auc_score(y_test, y_prob),
            'conf_matrix': confusion_matrix(y_test, y_pred)
        }

    def save(self):
        self.rf.save()
        self.lstm.save()
        print("✅ Hybrid model saved!")

    def load(self):
        self.rf.load()
        self.lstm = LSTMIDS(1)   # dim set on load
        self.lstm.load()