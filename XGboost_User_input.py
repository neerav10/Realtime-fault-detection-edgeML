
# ----------------------USER INPUT-------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Step 1: Load Dataset
file_path = r"C:\Users\neera\Desktop\MLLL_PROJECT\labeled_training_data(1).csv"
data = pd.read_csv(file_path)
print("Detected Columns:", data.columns.tolist())

# Step 2: Define Features and Targets
feature_cols = [
    'TEMP_1', 'PPM_1', 'LEVEL_1',
    'TEMP_2', 'PPM_2', 'LEVEL_2',
    'TEMP_3', 'PPM_3', 'LEVEL_3',
    'TEMP_4', 'PPM_4', 'LEVEL_4',
    'TEMP_5', 'PPM_5', 'LEVEL_5'
]
target_cols = ['gas_leak', 'temp_overshoot', 'tank_leak']

missing = [c for c in feature_cols + target_cols if c not in data.columns]
if missing:
    raise KeyError(f"Missing columns in dataset: {missing}")

# Step 3: Data Cleaning
for col in feature_cols + target_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.dropna(subset=feature_cols + target_cols, inplace=True)

# Step 4: Create Multi-Class Target
data['combined_label'] = data[target_cols].astype(str).agg(''.join, axis=1)
label_encoder = LabelEncoder()
data['encoded_label'] = label_encoder.fit_transform(data['combined_label'])

# Step 5: Prepare Data
X = data[feature_cols]
y = data['encoded_label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Train XGBoost Model
print(f"Training XGBoost for {len(label_encoder.classes_)} classes...")
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    random_state=42,
    n_estimators=200,         # Number of trees (estimators)
    learning_rate=0.05,       # Step size (learning rate)
    max_depth=6         # Maximum depth of trees
)

model.fit(X_train, y_train)

# Step 7: Predictions and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Performance")
print(f"Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClass Mappings:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i} → {cls} (gas_leak={cls[0]}, temp_overshoot={cls[1]}, tank_leak={cls[2]})")

# Condition names for interpretation
condition_names = ['gas_leak', 'temp_overshoot', 'tank_leak']

def interpret_prediction(binary_targets):
    descriptions = []
    for cond_name, val in zip(condition_names, binary_targets):
        if val == 1:
            descriptions.append(cond_name)
    if descriptions:
        return ", ".join(descriptions)
    else:
        return "No Fault Detected"

# Function to predict unseen data
def predict_unseen_xgb(new_data):
    new_data_scaled = scaler.transform(new_data)
    preds_encoded = model.predict(new_data_scaled)
    preds_decoded = label_encoder.inverse_transform(preds_encoded)
    preds_binary = [list(map(int, list(label))) for label in preds_decoded]
    return list(zip(preds_decoded, preds_binary))

# Function to get user input for unseen data prediction
def get_user_input(feature_names):
    print("Enter the values for the following features separated by commas:")
    print(", ".join(feature_names))
    input_str = input("Input: ")
    try:
        values = [float(x) for x in input_str.split(",")]
        if len(values) != len(feature_names):
            raise ValueError("Incorrect number of feature values.")
        user_df = pd.DataFrame([values], columns=feature_names)
        return user_df
    except Exception as e:
        print(f"Error: {e}")
        return None

# Interactive unseen data prediction with condition description
if __name__ == "__main__":
    feature_names = list(X.columns)
    new_data = get_user_input(feature_names)
    if new_data is not None:
        pred_result = predict_unseen_xgb(new_data)
        binary_output = pred_result[0][1]
        condition_text = interpret_prediction(binary_output)
        print(f"Predicted fault condition codes: {pred_result[0][0]}")
        print(f"Decoded binary output: {binary_output}")
        print(f"Condition(s) detected: {condition_text}")
    else:
        print("Invalid input, please try again.")
