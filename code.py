#import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read in dataset for mastercard igs score as a dataframe.
df_igs = pd.read_csv('igs_merged_c.csv')

#read in data on salary.
df_nsal = pd.read_csv('salary_norfolk.csv')
#df_ssal = 

#read in affordability data.
df_naff = pd.read_csv('Norfolk_Affordability_clean.csv')
df_saff = pd.read_csv('Portsmouth_Affordability_clean.csv')

#import mental health data
df = pd.read_csv('filtered_mental_health_data.csv')
print(df.head())

