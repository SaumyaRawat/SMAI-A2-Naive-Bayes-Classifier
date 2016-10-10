#!/bin/python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode

def read_data(filename):
	data = pd.read_csv(filename, sep=",", header = None)
	data.columns = ["age","class_of_worker","industry_code","occupation_code","education","wage_per_hour","enrolled","marital_status","major_industry_code","major_occupation_code","race","hispanic_origin","sex","labor_union","reason_for_unemployment","employment_stat","capital_gains","capital_losses","tax_liability","tax_filer_status","region_prev_residence","state_prev_residence","household_family_stat","household_summary","instance_weight","mc_msa","mc_reg","mc_within_reg","house_1","migration_prev_res","num_persons_worked_for_employer","fam_under_18","TPE","country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship","OB_or_SE","veterans_admin","veterans_benefits","weeks_worked_year","class"]
	return data

def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded
 
 #Create a new function:
def num_missing(x):
  return sum(x.isnull())


if __name__ == "__main__":
     train_data = read_data("dataset/census-income.data")
     test_data = read_data("dataset/census-income.data")

    #Applying per column:
     #print(train_data.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column

    #Binning ages
     #plt.hist(list(train_data['age']), facecolor='green', alpha=0.75, bins=9)
     bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
     group_names = [1,2,3,4,5,6,7,8,9]
     categories = pd.cut(train_data['age'], bins, labels=group_names)
     train_data['age'] = pd.cut(train_data['age'], bins, labels=group_names)
     
     categories = pd.cut(test_data['age'], bins, labels=group_names)
     test_data['age'] = pd.cut(test_data['age'], bins, labels=group_names)

     #Binning wage per hour
     #train_data.hist(column="wage_per_hour",bins=30)
     #group_names = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
     #train_data['wage_per_hour'] = pd.qcut(train_data['wage_per_hour'], 30, labels=group_names)
     
     #Binning capital_gains
     #train_data.hist(column="capital_gains",bins=30)
     #group_names = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
     #train_data['capital_gains'] = pd.qcut(train_data['capital_gains'], 30, labels=group_names)
     
    #Nominal labels to numeric values
     cat_columns = train_data.select_dtypes(['object']).columns
     train_data[cat_columns] = train_data[cat_columns].apply(lambda x: x.astype('category'))
     train_data[cat_columns] = train_data[cat_columns].apply(lambda x: x.cat.codes)

     test_data[cat_columns] = test_data[cat_columns].apply(lambda x: x.astype('category'))
     test_data[cat_columns] = test_data[cat_columns].apply(lambda x: x.cat.codes)
     #Coding LoanStatus as Y=1, N=0:
     #train_data[cat_columns] = coding(train_data[cat_columns], {'N':0,'Y':1})

     train_data.to_csv("dataset/train_data.csv", sep=',')
     test_data.to_csv("dataset/test_data.csv", sep=',')