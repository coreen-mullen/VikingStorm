
#import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Read in dataset for Mastercard IGS score as a DataFrame.
df_igs = pd.read_csv('igs_merged_c.csv')

# Read in data on salary.
df_nsal = pd.read_csv('df_nsal.csv', usecols=['STATE', 'Base Salary'])
df_nsal['Base Salary'] = df_nsal['Base Salary'].str.replace(',', '').astype(np.float32)

# Import mental health data
df_health = pd.read_csv('filtered_mental_health_data.csv', usecols=['STATE', 'Data_Value'])

# Calculate affordability score for Norfolk
df_n = pd.read_csv('Norfolk_Affordability_clean.csv', usecols=['STATE', 'hh_type1_h', 'hh_type1_t', 'commuter_1', 'avg_hh_s_1', 'blkgrp_med'])
df_n['affordability_score'] = (
    (df_n['hh_type1_h'] + df_n['hh_type1_t'] + (df_n['commuter_1'] / df_n['avg_hh_s_1'])) 
    / df_n['blkgrp_med'])
df_n['STATE'] = df_n['STATE'].replace(51, 710)
df_n.to_csv('updated_naff.csv', index=False)

# For Portsmouth
df_s = pd.read_csv('Norfolk_Affordability_clean.csv', usecols=['STATE', 'hh_type1_h', 'hh_type1_t', 'commuter_1', 'avg_hh_s_1', 'blkgrp_med'])
df_s['affordability_score'] = (
    (df_s['hh_type1_h'] + df_s['hh_type1_t'] + (df_s['commuter_1'] / df_s['avg_hh_s_1'])) 
    / df_s['blkgrp_med'])
df_s['STATE'] = df_s['STATE'].replace(51, 710)
df_s.to_csv('updated_saff.csv', index=False)

# Read in updated affordability files
df_n_1 = pd.read_csv('updated_naff.csv', usecols=['STATE', 'affordability_score'])
df_s_1 = pd.read_csv('updated_saff.csv', usecols=['STATE', 'affordability_score'])

# Convert STATE to int32
df_n_1['STATE'] = df_n_1['STATE'].astype(np.int32)
df_s_1['STATE'] = df_s_1['STATE'].astype(np.int32)

# Keeping necessary columns
df_nsal = df_nsal[['STATE', 'Base Salary']]
df_health = df_health[['STATE', 'Data_Value']]
print("Original df_n_1 NaN counts:\n", df_n_1.isna().sum())
print("Original df_health NaN counts:\n", df_health.isna().sum())
print("Original df_nsal NaN counts:\n", df_nsal.isna().sum())

# Merge Norfolk data with health data
features_norfolk = df_n_1.merge(df_health, on='STATE', how='left')

# Merge Portsmouth data with salary data
features_portsmouth = df_s_1.merge(df_nsal, on='STATE', how='left')

features_combined = pd.concat([features_norfolk, features_portsmouth], ignore_index=True)
# Keeping necessary columns
df_nsal = df_nsal[['STATE', 'Base Salary']]
df_health = df_health[['STATE', 'Data_Value']]# Check unique STATE values
print("Unique STATE values in df_n_1:", df_n_1['STATE'].unique())
print("Unique STATE values in df_health:", df_health['STATE'].unique())
print("Unique STATE values in df_nsal:", df_nsal['STATE'].unique())

# Merging with outer join to retain more data
features_norfolk = df_n_1.merge(df_health, on='STATE', how='outer')
features_portsmouth = df_s_1.merge(df_nsal, on='STATE', how='outer')

# Combine features
features_combined = pd.concat([features_norfolk, features_portsmouth], ignore_index=True)

# Check NaN counts after merging
print("NaN counts after merging:\n", features_combined.isna().sum())

#fill missing values with average of 
features_combined.fillna(features_combined.mean(), inplace=True)

#scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_combined[['affordability_score', 'Data_Value', 'Base Salary']])

#perform clustering
n_clusters = 3  # Choose the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
features_combined['cluster'] = kmeans.fit_predict(X_scaled)

#inspect the clusters
print(features_combined)
#plot graphs

X = features_combined[['affordability_score', 'cluster']]
y = features_combined['Data_Value']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Feature importance
importance = rf_model.feature_importances_
print(f'Feature Importances: {importance}')