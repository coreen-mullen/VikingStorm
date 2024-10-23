# VikingStorm

## Table of Contents
1. [Background and Description](#background-and-description)
2. [Data and Preprocessing](#data-and-preprocessing)
    - [Data Sources](#data-sources)
    - [Data Description](#data-description)
3. [Code](#code)
4. [Resources](#resources)

---

## VikingStorm
This project is part of the 2024 Mastercard Center for Inclusive Growth x AUC DSI Data Challenge.
  - In this project we work with the Mastercard Inclusive Growth Score ([IGS]https://inclusivegrowthscore.com) to perform analysis on the relationship between various socioeconmic factors on mental health in Portsmouth and Norfolk, Va.
  - Our project looks at these two counties and looks at the reasoning for lower scores in certain Inclusive Growth score subgroups.
---

## Background and Description

### Research Question 
What are the effects of socioeconomic factors such as income and housing affordability on mental health?

---

## Data and Preprocessing

### Data Sources
- [IGS Data](https://inclusivegrowthscore.com/)
- [Affordability Score](https://hudgis-hud.opendata.arcgis.com/datasets/HUD::location-affordability-index-v-2-0/about)
- [Employee Salary - Norfolk](https://data.norfolk.gov/Government/Employee-Salaries/4fsk-z8s8/data)

### Data Description
The IGS dataset is filtered for Norfolk and Portsmouth Countys only at the Urban-Rural level.
The Affordability Score dataset was filtered down to include data on both citys seperately.
Employee Salary data is used for our income comparisons, this data is only available for Norfolk currently.

---
## Code
The code for this project comes in the form of a linear regression code, a k-means clustering algorithm ( in code_new.py ), and a Random Forest Regression (in code_testing.py).

The "affordability_score" used in the clustering and random forest regression were calculated from this formula 
![image](https://github.com/user-attachments/assets/302fe64f-ef3b-4328-ad65-a5f70cb0ba0f)


---
## Resources
- [Paper for Research](https://archive.cdc.gov/#/details?url=https://www.cdc.gov/hrqol/pdfs/mhd.pdf )
