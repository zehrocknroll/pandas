# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 23:22:55 2022

@author: zehra
"""


#import pandas

import pandas as pd


#pandas series

s = pd.Series([1,6,9,12,53])
type(s) #pandas.core.series.Series

s.index # RangeIndex(start=0, stop=5, step=1)

s.dtype #dtype('int64')

s.size #5

s.ndim #1

s.values #array([ 1,  6,  9, 12, 53], dtype=int64)

type(s.values) #numpy.ndarray

s.head(3)

#0    1
#1    6
#2    9
#dtype: int64


#reading data


"""data = pd.read_csv("advertising.csv")"""


"data.head()"

"""
    TV  radio  newspaper  sales
1  230.1   37.8       69.2   22.1
2   44.5   39.3       45.1   10.4
3   17.2   45.9       69.3    9.3
4  151.5   41.3       58.5   18.5
5  180.8   10.8       58.4   12.9

"""


# quick look at data

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic1")

"""
       survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0           0       3    male  22.0  ...   NaN  Southampton     no  False
1           1       1  female  38.0  ...     C    Cherbourg    yes  False
2           1       3  female  26.0  ...   NaN  Southampton    yes   True
3           1       1  female  35.0  ...     C  Southampton    yes  False
4           0       3    male  35.0  ...   NaN  Southampton     no   True
..        ...     ...     ...   ...  ...   ...          ...    ...    ...
886         0       2    male  27.0  ...   NaN  Southampton     no   True
887         1       1  female  19.0  ...     B  Southampton    yes   True
888         0       3  female   NaN  ...   NaN  Southampton     no  False
889         1       1    male  26.0  ...     C    Cherbourg    yes   True
890         0       3    male  32.0  ...   NaN   Queenstown     no   True

[891 rows x 15 columns]

"""

df.tail()

"""
       survived  pclass     sex   age  ...  deck  embark_town  alive  alone
886         0       2    male  27.0  ...   NaN  Southampton     no   True
887         1       1  female  19.0  ...     B  Southampton    yes   True
888         0       3  female   NaN  ...   NaN  Southampton     no  False
889         1       1    male  26.0  ...     C    Cherbourg    yes   True
890         0       3    male  32.0  ...   NaN   Queenstown     no   True

[5 rows x 15 columns]

"""

df.shape #(891, 15)

df.info()


"""
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB

"""

df.columns

"""
Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
       'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town',
       'alive', 'alone'],
      dtype='object')

"""

df.index # RangeIndex(start=0, stop=891, step=1)


df.describe().T

"""
  count       mean        std   min      25%      50%   75%       max
survived  891.0   0.383838   0.486592  0.00   0.0000   0.0000   1.0    1.0000
pclass    891.0   2.308642   0.836071  1.00   2.0000   3.0000   3.0    3.0000
age       714.0  29.699118  14.526497  0.42  20.1250  28.0000  38.0   80.0000
sibsp     891.0   0.523008   1.102743  0.00   0.0000   0.0000   1.0    8.0000
parch     891.0   0.381594   0.806057  0.00   0.0000   0.0000   0.0    6.0000
fare      891.0  32.204208  49.693429  0.00   7.9104  14.4542  31.0  512.3292

"""

df.isnull().values.any() #True

df.isnull().sum()



df["sex"].head()
"""
0      male
1    female
2    female
3    female
4      male
Name: sex, dtype: object
"""




df["sex"].value_counts()
"""
male      577
female    314
Name: sex, dtype: int64
"""


#selection in pandas


import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic1")

df.head()

"""
   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True

[5 rows x 15 columns]
"""

df.index #RangeIndex(start=0, stop=891, step=1)

df[0:13]

"""
   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0          0       3    male  22.0  ...   NaN  Southampton     no  False
1          1       1  female  38.0  ...     C    Cherbourg    yes  False
2          1       3  female  26.0  ...   NaN  Southampton    yes   True
3          1       1  female  35.0  ...     C  Southampton    yes  False
4          0       3    male  35.0  ...   NaN  Southampton     no   True
5          0       3    male   NaN  ...   NaN   Queenstown     no   True
6          0       1    male  54.0  ...     E  Southampton     no   True
7          0       3    male   2.0  ...   NaN  Southampton     no  False
8          1       3  female  27.0  ...   NaN  Southampton    yes  False
9          1       2  female  14.0  ...   NaN    Cherbourg    yes  False
10         1       3  female   4.0  ...     G  Southampton    yes  False
11         1       1  female  58.0  ...     C  Southampton    yes   True
12         0       3    male  20.0  ...   NaN  Southampton     no   True

[13 rows x 15 columns]
"""

df.drop(0,axis =0).head()
"""
   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True
5         0       3    male   NaN  ...   NaN   Queenstown     no   True

[5 rows x 15 columns]
"""


delete_indexes = [1,6,9,10]

df.drop(delete_indexes,axis =0).head()
"""
 survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True
5         0       3    male   NaN  ...   NaN   Queenstown     no   True

[5 rows x 15 columns]
"""

# df.drop(delete_indexes,axis =0, inplace=True) : make this change permanent


#convert variable to index


df["age"].head()
df.age.head()


df.index = df["age"]

df.drop("age",axis = 1).head()
df.head()

df.drop("age",axis = 1, inplace = True)
df.head()



#convert index to variable


df.index

df["age"] = df.index
df.head()

df.drop("age",axis = 1, inplace = True)


df.reset_index().head()

df.reset_index()


a = df.head()


#operations on variables


import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None) #to get rid of the three dots in the printouts
df = sns.load_dataset("titanic1")
df.head()


"age" in df
#True

df["age"].head()

type(df["age"].head())
#pandas.core.series.Series


df[["age"]].head()


type(df[["age"]].head())
#pandas.core.frame.DataFrame


df[["age","alive"]]


col_names = ["age","adult_male", "alive"]
df[col_names]



df["age2"] = df["age"]**2



df.drop("age2",axis = 1).head()



#iloc & loc

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic1")
df.head()


#iloc: integer based selection
df.iloc[0:3]
df.iloc[0,0]



#loc: label based selection
df.loc[0:3]


df.iloc[0:3, 0:3]


df.loc[0:3, "age"]



col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]




#Conditional Selection

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic1")
df.head()

df[df["age"] > 50].head() #those over the age of 50

df[df["age"] > 50]["age"].count() #Number of people over the age of 50

df.loc[df["age"] > 50, "class"].head()

df.loc[df["age"] > 50, ["age","class"]].head()

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age","class"]].head()


df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50) 
        &(df["sex"] == "male") 
        &((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
        ["age","class","embark_town"]]

df_new["embark_town"].value_counts()



#Aggregation & Grouping


import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic1")
df.head()



df["age"].mean()

df.groupby("sex")["age"].mean()


df.groupby("sex").agg({"age":"mean"})


df.groupby("sex").agg({"age":["mean","sum"]})



df.groupby("sex").agg({"age":["mean","sum"],
                       "survived":"mean"})





df.groupby(["sex","embark_town","class"]).agg({
    "age":["mean"],
    "survived":"mean",
    "sex": "count"})




#pivot table



import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic1")
df.head()


df.pivot_table("survived","sex","embarked")


df.pivot_table("survived","sex","embarked",aggfunc="std")














