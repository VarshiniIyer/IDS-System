import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Load dataset - replace with your CSV filename
data = pd.read_csv('processed_train.csv')

# Convert columns to numeric and fill NaNs (example for your dataset)
data['duration'] = pd.to_numeric(data['duration'], errors='coerce').fillna(0)
data['dst_host_srv_rerror_rate'] = pd.to_numeric(data['dst_host_srv_rerror_rate'], errors='coerce').fillna(0)

# Categorical columns to encode
categorical_cols = ['protocol_type', 'service', 'flag']

# Dictionary to store encoders for each categorical feature
encoders = {}

# Encode categorical features and save encoders
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le
    with open(f'{col}_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

# Encode label column and save encoder
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'].astype(str))
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Data types after preprocessing:")
print(data.dtypes)

# Prepare features and target
X = data.drop('label', axis=1)
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost classifier
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Training completed successfully. Model and encoders saved.")
