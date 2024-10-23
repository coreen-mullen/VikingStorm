#import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#int32 conversion is used on multiple datasets to take less memory.
# read in dataset for mastercard igs score as a dataframe.
df_igs = pd.read_csv('igs_merged_c.csv')

#read in data on salary.
df_nsal = pd.read_csv('df_nsal.csv', usecols=['STATE', 'Base Salary'])
df_nsal['Base Salary']=df_nsal['Base Salary'].str.replace(',', '').astype(np.float32)
#df_ssal = 

#import mental health data
df_health = pd.read_csv('filtered_mental_health_data.csv', usecols=['STATE','Data_Value'])
df_health['Data_Value'] = df_health['Data_Value']
#calculate affordability score
#for norfolk
df_n = pd.read_csv('Norfolk_Affordability_clean.csv',usecols=['STATE', 'hh_type1_h', 'hh_type1_t', 'commuter_1', 'avg_hh_s_1', 'blkgrp_med'])
df_n['affordability_score'] = (
    (df_n['hh_type1_h'] + df_n['hh_type1_t'] + (df_n['commuter_1'] / df_n['avg_hh_s_1'])) 
    / df_n['blkgrp_med'])
df_n['STATE'] = df_n['STATE'].replace(51, 710)
df_n.to_csv('updated_naff.csv', index=False)

#for portsmouth
df_s = pd.read_csv('Norfolk_Affordability_clean.csv',usecols=['STATE', 'hh_type1_h', 'hh_type1_t', 'commuter_1', 'avg_hh_s_1', 'blkgrp_med'])
df_s['affordability_score'] = (
    (df_s['hh_type1_h'] + df_s['hh_type1_t'] + (df_s['commuter_1'] / df_s['avg_hh_s_1'])) 
    / df_s['blkgrp_med'])
df_s['STATE'] = df_s['STATE'].replace(51, 710)
df_s.to_csv('updated_saff.csv', index=False)

#read in updated affordability file.
df_n_1 = pd.read_csv('updated_naff.csv',usecols=['STATE', 'affordability_score'])
df_s_1 = pd.read_csv('updated_saff.csv',usecols=['STATE', 'affordability_score'])

df_n_1['STATE'] = df_n_1['STATE'].astype(np.int32)
df_s_1['STATE'] = df_s_1['STATE'].astype(np.int32)
#keeping necessary columns
df_nsal = df_nsal[['STATE', 'Base Salary']]
df_health = df_health[['STATE', 'Data_Value']]
df_n_1 = df_n_1[['STATE','affordability_score']]
df_s_1 = df_s_1[['STATE','affordability_score']]


# merge the data and clean missing values
df = pd.merge(df_n_1, df_s_1, on='STATE', how='inner')

df = pd.merge(df, df_health, on='STATE', how='inner')

df = pd.merge(df, df_nsal, on='STATE', how='inner')
df.dropna(inplace=True)
print(df.shape())

# Step 3: Select relevant columns for clustering (e.g., income, housing affordability, mental health)
# Adjust these columns according to your dataset
X = df[['Data_Value', 'Base Salary', 'affordability_score']]
#Where Data_Value represents mental health score ( Higher score is negative in relation to mental health.).


# Step 4: Standardize the data (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply K-Means Clustering
# Define the number of clusters (you can try different values of 'k' and use elbow method to find the best one)
kmeans = KMeans(n_clusters=3, random_state=42)  # Example with 3 clusters
df['cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Visualize the clusters
# Create a scatter plot of two features, colored by cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['income'], y=df['housing_cost'], hue=df['cluster'], palette='viridis', s=100)
plt.title('Clusters of Locations based on Income and Housing Cost')
plt.xlabel('Income')
plt.ylabel('Housing Cost')
plt.show()

# Step 7: Evaluate and interpret clusters
# You can explore the cluster means to understand the characteristics of each cluster
cluster_means = df.groupby('cluster').mean()
print(cluster_means)

