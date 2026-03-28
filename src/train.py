import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              roc_auc_score, confusion_matrix)

print("=" * 50)
print("   🚀 STARTING MODEL TRAINING")
print("=" * 50)

# ── Step 1: Load Processed Data ──
print("\n📦 Loading processed data...")

X_train = np.load('data/processed/X_train.npy')
X_test  = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test  = np.load('data/processed/y_test.npy')

print(f"   X_train : {X_train.shape}")
print(f"   X_test  : {X_test.shape}")
print(f"   y_train : {y_train.shape}")
print(f"   y_test  : {y_test.shape}")

# ── Step 2: Reduce data if RAM is low ──
print("\n✂️  Reducing data for low RAM PC...")
from sklearn.model_selection import train_test_split

X_train, _, y_train, _ = train_test_split(
    X_train, y_train,
    test_size    = 0.5,      # use 50% of training data
    random_state = 42,
    stratify     = y_train
)
print(f"   Reduced X_train : {X_train.shape}")

# ── Step 3: Train Random Forest ──
print("\n🌲 Training Random Forest...")
print("   Please wait — this may take 2-5 minutes...")

rf = RandomForestClassifier(
    n_estimators = 30,       # low for RAM safety
    max_depth    = 8,
    random_state = 42,
    n_jobs       = 1,        # single core for low RAM
    verbose      = 1         # shows progress
)

rf.fit(X_train, y_train)
print("   ✅ Training complete!")

# ── Step 4: Evaluate ──
print("\n📊 Evaluating model...")
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 50)
print("   📈 RESULTS")
print("=" * 50)
print(f"   Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"   Precision : {prec:.4f} ({prec*100:.1f}%)")
print(f"   Recall    : {rec:.4f}  ({rec*100:.1f}%)")
print(f"   F1 Score  : {f1:.4f}  ({f1*100:.1f}%)")
print(f"   ROC-AUC   : {auc:.4f}  ({auc*100:.1f}%)")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n   Confusion Matrix:")
print(f"   TN={cm[0][0]}  FP={cm[0][1]}")
print(f"   FN={cm[1][0]}  TP={cm[1][1]}")

# ── Step 5: Save Model ──
print("\n💾 Saving model...")
os.makedirs('models', exist_ok=True)

with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f,
                protocol=pickle.HIGHEST_PROTOCOL)

# ── Step 6: Verify file saved correctly ──
size = os.path.getsize('models/rf_model.pkl')
print(f"   File size : {size/1024:.1f} KB")

if size > 100:
    print("   ✅ rf_model.pkl saved successfully!")
else:
    print("   ❌ File too small — something went wrong!")
    exit()

# ── Step 7: Test loading model back ──
print("\n🔁 Testing model reload...")
with open('models/rf_model.pkl', 'rb') as f:
    rf_loaded = pickle.load(f)

test_pred = rf_loaded.predict(X_test[:5])
print(f"   Test predictions : {test_pred}")
print("   ✅ Model loads correctly!")

# ── Done ──
print("\n" + "=" * 50)
print("   ✅ ALL DONE! READY TO LAUNCH DASHBOARD")
print("=" * 50)
print("\n   Run this next:")
print("   streamlit run app.py")