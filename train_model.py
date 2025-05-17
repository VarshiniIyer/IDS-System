import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the processed data
train_df = pd.read_csv('processed_train.csv')
test_df = pd.read_csv('processed_test.csv')

# Load encoders
with open('protocol_type_encoder.pkl', 'rb') as f:
    protocol_encoder = pickle.load(f)
with open('service_encoder.pkl', 'rb') as f:
    service_encoder = pickle.load(f)
with open('flag_encoder.pkl', 'rb') as f:
    flag_encoder = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Encode categorical columns in train data
train_df['protocol_type'] = protocol_encoder.transform(train_df['protocol_type'])
train_df['service'] = service_encoder.transform(train_df['service'])
train_df['flag'] = flag_encoder.transform(train_df['flag'])
train_df['label'] = label_encoder.transform(train_df['label'])

# Encode categorical columns in test data
test_df['protocol_type'] = protocol_encoder.transform(test_df['protocol_type'])
test_df['service'] = service_encoder.transform(test_df['service'])
test_df['flag'] = flag_encoder.transform(test_df['flag'])
test_df['label'] = label_encoder.transform(test_df['label'])

# Separate features and label
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and columns
with open('ids_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

print("\nModel training complete! Saved as ids_model.pkl and columns.pkl")
