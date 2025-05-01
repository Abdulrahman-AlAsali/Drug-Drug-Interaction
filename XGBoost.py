import os
import pandas as pd
import time
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------
# Configuration
# --------------------------------------
DATA_DIR = "./DDI_data_196_drugs"
ddi_tables = []  # Adjust based on your CSVs

# --------------------------------------
# Ingest CSV files
# --------------------------------------
count = 0
print(f"found {len(os.listdir(DATA_DIR))} files in {DATA_DIR}")
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".csv"):
        print(f"Processing file: {filename}")
        table_name = filename.replace(".csv", "")
        ddi_tables.append(table_name)
        file_path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(file_path)
        
        # check for temp dir 
        if not os.path.exists("temp/ingest"):
            os.makedirs("temp/ingest")

        # Save to temp directory
        temp_path = os.path.join("temp/ingest", f"{table_name}.pkl")
        df.to_pickle(temp_path)  # Save to temp directory
        print(f"Ingested {filename} as `{table_name}.pkl`")
        count += 1
print(f"Total files processed: {count}")

# --------------------------------------
# Transform data
# --------------------------------------
for table in ddi_tables:
    print(f"Transforming table: {table}")
    file_path = os.path.join("temp/ingest", f"{table}.pkl")
    df = pd.read_pickle(file_path)

    if table == "top196drugs":
        original_col_name = df.columns[0]
        df.columns = ['drug_name']
        df.loc[-1] = [original_col_name]
        df.index = df.index + 1  
        df = df.sort_index()  
        df.reset_index(drop=True, inplace=True)
        df.insert(0, 'drug_id', range(1, len(df) + 1))
    else:
        df.index = range(1, len(df) + 1)
        df.columns = range(1, len(df.columns) + 1)
        df = df.stack().reset_index()
        df.columns = ['first_drug_id', 'second_drug_id', 'interaction_value']
        df = df[df['first_drug_id'] <= df['second_drug_id']]

            # check for temp dir 
    if not os.path.exists("temp/transformed"):
        os.makedirs("temp/transformed")

    # Save to temp directory
    temp_path = os.path.join("temp/transformed", f"{table_name}.pkl")
    df.to_pickle(temp_path)
    print(f"Saved transformed table: {table}_transformed.pkl")

# --------------------------------------
# Load and merge transformed data
# --------------------------------------
tranformed_path = os.path.join("temp/transformed", "truelabel_196_transformed.pkl")
labels_df = pd.read_pickle(tranformed_path)
feature_tables = [t + "_transformed" for t in ddi_tables if t not in ["truelabel_196", "top196drugs"]]

merged = labels_df.copy()
for table in feature_tables:
    tranformed_path = os.path.join("temp/transformed", f"{table}.pkl")
    df = pd.read_pickle(tranformed_path)
    df = df.rename(columns={"interaction_value": f"{table}_score"})
    merged = pd.merge(merged, df, on=["first_drug_id", "second_drug_id"], how="left")

merged.dropna(inplace=True)

# --------------------------------------
# Train model
# --------------------------------------
X = merged.drop(columns=["interaction_value", "first_drug_id", "second_drug_id"])
y = merged["interaction_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist, 
    n_iter=20,
    scoring="accuracy", 
    cv=3, 
    verbose=1, 
    random_state=42,
    n_jobs=-1
)

start = time.time()
search.fit(X_train, y_train)
end = time.time()

best_model = search.best_estimator_

# Calibrate the model
calibrated = CalibratedClassifierCV(best_model, method='sigmoid', cv=3)
calibrated.fit(X_train, y_train)
y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
brier = brier_score_loss(y_test, y_prob_cal)


acc = calibrated.score(X_test, y_test)
# Get predicted probabilities for the positive class (label 1)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate Brier score
brier = brier_score_loss(y_test, y_prob)

print(f"Brier score on test set: {brier:.4f}")
print(f"Best parameters: {search.best_params_}")
print(f"Accuracy on test set: {acc:.4f}")
print(f"Training time: {end - start:.2f} seconds")

# Feature importances
feature_importance = best_model.feature_importances_
feature_names = X.columns
importance_dict = dict(zip(feature_names, feature_importance))

print("Top features:")
for k, v in sorted(importance_dict.items(), key=lambda x: -x[1])[:10]:
    print(f"{k}: {v:.4f}")


# ---------------------------
# Visualizations
# ---------------------------

# 1. Feature Importance Plot
sorted_idx = np.argsort(feature_importance)[::-1]
length = len(sorted_idx)
plt.figure(figsize=(length, 6))
plt.bar(range(length), feature_importance[sorted_idx], align='center')
plt.xticks(range(length), np.array(feature_names)[sorted_idx], rotation=45, ha='right')
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()

# 2. Histogram of Predicted Probabilities
plt.figure(figsize=(8, 5))
plt.hist(y_prob, bins=20, edgecolor='k')
plt.xlabel("Predicted Probability for the interaction class")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Probabilities")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Confusion Matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()