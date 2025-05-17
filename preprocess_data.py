import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Define column names as per NSL-KDD
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

# Load datasets
train_df = pd.read_csv("KDDTrain+.txt", names=column_names)
test_df = pd.read_csv("KDDTest+.txt", names=column_names)

print("📋 Columns loaded:")
print(train_df.columns.tolist())

# Encode categorical columns safely
categorical_features = ['protocol_type', 'service', 'flag']

for feature in categorical_features:
    print(f"\n🔤 Encoding {feature}...")

    le = LabelEncoder()
    train_df[feature] = le.fit_transform(train_df[feature])
    joblib.dump(le, f"{feature}_encoder.pkl")

    # Create a mapping dictionary
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # Handle unseen categories in test set
    test_df[feature] = test_df[feature].apply(lambda x: mapping.get(x, -1))  # -1 for unseen
    print(f"Encoded {feature}. Unseen test values assigned -1.")

# Encode label column
print("\n🎯 Encoding label column...")
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])
joblib.dump(label_encoder, "label_encoder.pkl")

# Save processed CSVs
train_df.to_csv("processed_train.csv", index=False)
test_df.to_csv("processed_test.csv", index=False)

print("\n✅ Preprocessing complete!")
print("Saved:")
print(" - processed_train.csv")
print(" - processed_test.csv")
print(" - protocol_type_encoder.pkl")
print(" - service_encoder.pkl")
print(" - flag_encoder.pkl")
print(" - label_encoder.pkl")
