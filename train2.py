import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Parameters
output_file = 'model_xgb.bin'

# Data preparation
print('Loading data...')
df = pd.read_csv('data/malaria_dataset.csv')

# Drop non-predictive columns
columns_to_drop = ['IP_Number', 'DOA', 'Discharge_Date', 
                   'Primary_Code', 'Diagnosis_Type', 'Risk_Score']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Encode categorical
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != 'Target':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Split data
print('Splitting data...')
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Prepare matrices
# Extract target using bracket notation
y_train = df_train['Target'].values
y_val = df_val['Target'].values
y_test = df_test['Target'].values

# Drop target column
df_train = df_train.drop('Target', axis=1)
df_val = df_val.drop('Target', axis=1)
df_test = df_test.drop('Target', axis=1)

# Feature matrices
features = df_train.columns.tolist()
X_train = df_train[features].values
X_val = df_val[features].values
X_test = df_test[features].values

# Train model
print('Training model...')
dfulltrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=100)

# Validation
y_pred = model.predict(dval)
auc = roc_auc_score(y_val, y_pred)
print(f'Validation AUC: {auc:.4f}')

# Test
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
y_test_pred = model.predict(dtest)
test_auc = roc_auc_score(y_test, y_test_pred)
print(f'Test AUC: {test_auc:.4f}')

# Save model
with open(output_file, 'wb') as f_out:
    pickle.dump((model, features), f_out)

print(f'Model saved to {output_file}')
