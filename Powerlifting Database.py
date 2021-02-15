#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from scipy.stats import ttest_ind, ttest_ind_from_stats

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


df_main = pd.read_csv(r'C:\Users\raflg\Downloads\Databases\openpowerlifting.csv', low_memory=False)
df_main.head()


# In[3]:


df_main.shape


# In[4]:


df_main.dtypes


# # Clean up the database

# In[5]:


df_main.isnull().sum()


# In[6]:


df_main['Sex'] = df_main['Sex'].astype('category')
df_main['Event'] = df_main['Event'].astype('category')
df_main['Equipment'] = df_main['Equipment'].astype('category')
df_main['AgeClass'] = df_main['AgeClass'].astype('category')
df_main['Division'] = df_main['Division'].astype('category')
df_main['WeightClassKg'] = df_main['WeightClassKg'].astype('category')
df_main['Place'] = df_main['Place'].astype('category')
df_main['Tested'] = df_main['Tested'].astype('category')
df_main['Country'] = df_main['Country'].astype('category')
df_main['Federation'] = df_main['Federation'].astype('category')
df_main['MeetCountry'] = df_main['MeetCountry'].astype('category')
df_main['MeetState'] = df_main['MeetState'].astype('category')
df_main['MeetName'] = df_main['MeetName'].astype('category')
df_main['Date'] = pd.to_datetime(df_main['Date'])


# In[7]:


df_main.info()


# In[8]:


df_main.duplicated().sum()
df_main.loc[df_main.duplicated(keep=False)]
df_main.drop_duplicates(keep='first', inplace=True)


# # Task 1: Ratio Total/Body weight in 1990's vs 2000's
# ### H0 : The ratio is the same in the 1990's and 2000's

# In[9]:


columns_filt = ['Name', 'Sex', 'Date', 'BodyweightKg', 'Equipment', 'MeetCountry', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']
df_ratio = df_main[columns_filt]


# In[10]:


df_ratio[['BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']] =     df_ratio[['BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']].fillna(0)


# In[11]:


df_ratio = df_ratio.loc[(df_ratio['BodyweightKg'] > 0) & (df_ratio['Best3SquatKg'] > 0) & (df_ratio['Best3BenchKg'] > 0) & 
                        (df_ratio['Best3DeadliftKg'] > 0)]


# In[12]:


df_ratio['TotalKg'] = df_ratio['Best3SquatKg'] + df_ratio['Best3BenchKg'] + df_ratio['Best3DeadliftKg']
df_ratio['Ratio'] = (df_ratio['TotalKg'] / df_ratio['BodyweightKg']).round(1)
df_ratio


# #### Let's see how the distribution looks like for each decade

# In[13]:


df_decades = df_ratio.loc[((df_ratio['Date'] >= '1990-01-01') & (df_ratio['Date'] < '2010-01-01'))]
                                   
df_decades['Period'] = np.where((df_decades['Date'] >= '1990-01-01') & (df_decades['Date'] < '2000-01-01'), '1990s', '2000s')
df_decades.groupby('Period')['Ratio'].describe().round(2)


# In[33]:


sns.catplot(data=df_decades, x='Period', y='Ratio', kind='boxen')


# In[14]:


ratio_1990 = df_decades[df_decades['Period'] == '1990s']
ratio_2000 = df_decades[df_decades['Period'] == '2000s']


# In[25]:


print(ratio_1990.shape)
print(ratio_2000.shape)


# In[17]:


balanced_2000 = ratio_2000.sample(ratio_1990.shape[0])


# In[36]:


def t_test ():
    stat, p = ttest_ind(ratio_1990['Ratio'], balanced_2000['Ratio'], equal_var = False)
    if p > 0.05:
        return 'p-value = {} and > to 0.05 so we have same distributions and we fail to reject H0'.format(p)
    else :
        return 'p-value = {} and < to 0.05 so we have different distributions and we reject H0'.format(p)


# In[37]:


t_test()


# ### Conclusion : The ratio is different in the 1990's and 2000's
# #### But the distributions seem close looking to the table above. Let's try an other t-test
# #### Maybe we can run a t-test for some events / Meet Countries: USA, Australia, and Russia

# In[20]:


def t_test2 (country):
    sample_one = ratio_1990[ratio_1990['MeetCountry'] == country]['Ratio']
    sample_two = ratio_2000[ratio_2000['MeetCountry'] == country]['Ratio']
    stat, p = ttest_ind_from_stats(mean1 = sample_one.mean(), std1 = sample_one.std(), nobs1 = sample_one.shape[0],
                                   mean2 = sample_two.mean(), std2 = sample_two.std(), nobs2 = sample_two.shape[0],
                                   equal_var = False)
    if p > 0.05:
        return 'p-value = {} and > to 0.05 so we have same distributions and we fail to reject H0'.format(p)
    else :
        return 'p-value = {} and < to 0.05 so we have different distributions and we reject H0'.format(p)


# In[21]:


for country in ['USA', 'Australia', 'Russia']:
    print(f'{country :-<12}{t_test2(country)}')


# ### Conclusion: The ratio was different in the 1990's and 2000's with a 95% confidence

# # Task 2: Compare Ratio over the years depending of the equipment 

# In[22]:


df_ratio['Equipment'].value_counts()
ratio_comparison = df_ratio.groupby([pd.Grouper(key='Date', freq='Y'), 'Equipment', 'Sex'])['Ratio'].median().reset_index()
ratio_comparison = ratio_comparison.loc[ratio_comparison['Equipment'] != 'Straps']
ratio_comparison['Ratio'] = ratio_comparison['Ratio'].fillna(0)
ratio_comparison['Date'] = ratio_comparison['Date'].dt.strftime('%Y')
ratio_comparison


# In[23]:


maxrange_y = ratio_comparison['Ratio'].max() + 1
px.bar(ratio_comparison, x='Equipment', y='Ratio', animation_frame = 'Date', facet_col = 'Sex', color ='Sex',
      opacity = 0.75, range_y=[0, maxrange_y])


# # Task 3: Compare how many athletes compete Raw over the years

# In[24]:


nb_athl_raw = df_main.groupby(pd.Grouper(key='Date', freq='Y')).agg(PCT=('Equipment', lambda x: (x=='Raw').sum()))
nb_athl_total = df_main.groupby(pd.Grouper(key='Date', freq='Y')).agg(PCT=('Name', 'count'))

nb_athl_raw_pct = (nb_athl_raw.div(nb_athl_total)*100).reset_index().round(1)
nb_athl_raw_pct['Date'] = nb_athl_raw_pct['Date'].dt.strftime('%Y')

px.line(nb_athl_raw_pct, x='Date', y='PCT')


# In[ ]:





# In[ ]:




