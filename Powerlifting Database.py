#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import scipy.stats as stats

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


# # Task 1: Compare the Ratio (Total/Body weight)
# ### H0 : The ratio is the same between 1990's and 2000's

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


# In[13]:


decades_cond = [(df_ratio['Date'] >= '1990-01-01') & (df_ratio['Date'] < '2000-01-01'),
                (df_ratio['Date'] >= '2000-01-01') & (df_ratio['Date'] < '2010-01-01')]

decades_res = ['1990s', '2000s']

df_ratio['Decade'] = np.select(decades_cond, decades_res)


# In[14]:


df_ratio_desc = df_ratio[(df_ratio['Decade'] == '1990s') | (df_ratio['Decade'] == '2000s')]
df_ratio_desc.groupby(['Decade', 'Sex'])['Ratio'].describe().round(2)


# In[15]:


sns.catplot(data=df_ratio_desc, x='Sex', y='Ratio', hue='Decade', kind='boxen', palette='pastel')


# In[16]:


df_ttest = pd.crosstab(df_ratio_desc['Name'], df_ratio_desc['Decade'], values=df_ratio_desc['Ratio'], aggfunc='max').dropna()
stats.ttest_rel(a=df_ttest['1990s'], b=df_ttest['2000s'], alternative='less')


# In[17]:


for sex in df_ratio_desc.Sex.unique():
    df_ttest2 = pd.crosstab(df_ratio_desc.loc[df_ratio_desc['Sex'] == sex]['Name'], 
                           df_ratio_desc.loc[df_ratio_desc['Sex'] == sex]['Decade'], 
                           values=df_ratio_desc.loc[df_ratio_desc['Sex'] == sex]['Ratio'], 
                           aggfunc='median').dropna()
    stat, p = stats.ttest_rel(a=df_ttest2['1990s'], b=df_ttest2['2000s'], alternative='less')
    print(p)


# In[18]:


stats.t.interval(alpha=0.95,
                 df=len(df_ttest['1990s'])-1,
                 loc=df_ttest['1990s'].mean(),
                 scale=df_ttest['1990s'].std(ddof=1))


# In[19]:


stats.t.interval(alpha=0.95,
                 df=len(df_ttest['2000s'])-1,
                 loc=df_ttest['2000s'].mean(),
                 scale=df_ttest['2000s'].std(ddof=1))


# In[77]:


ct = pd.crosstab([df_ratio_desc['Name'], df_ratio_desc['Sex']], df_ratio_desc['Decade'], 
            values=df_ratio_desc['Ratio'], aggfunc='max').reset_index().dropna()

ct_arranged = ct.melt(id_vars=['Name','Sex'], var_name='Decade', value_name='Ratio')

sns.catplot(data=ct_arranged, x='Sex', y='Ratio', hue='Decade', kind='boxen', palette='GnBu')


# ### Conclusion: The Ratio is different betwenn 1990s and 2000s

# # Task 2: Compare Ratio over the years depending of the equipment 

# In[20]:


df_ratio['Equipment'].value_counts()
ratio_comparison = df_ratio.groupby([pd.Grouper(key='Date', freq='Y'), 'Equipment', 'Sex'])['Ratio'].median().reset_index()
ratio_comparison = ratio_comparison.loc[ratio_comparison['Equipment'] != 'Straps']
ratio_comparison['Ratio'] = ratio_comparison['Ratio'].fillna(0)
ratio_comparison['Date'] = ratio_comparison['Date'].dt.strftime('%Y')
ratio_comparison


# In[21]:


maxrange_y = ratio_comparison['Ratio'].max() + 1
px.bar(ratio_comparison, x='Equipment', y='Ratio', animation_frame = 'Date', facet_col = 'Sex', color ='Sex',
      opacity = 0.75, range_y=[0, maxrange_y])


# # Task 3: Compare how many athletes compete Raw over the years

# In[22]:


nb_athl_raw = df_main.groupby(pd.Grouper(key='Date', freq='Y')).agg(PCT=('Equipment', lambda x: (x=='Raw').sum()))
nb_athl_total = df_main.groupby(pd.Grouper(key='Date', freq='Y')).agg(PCT=('Name', 'count'))

nb_athl_raw_pct = (nb_athl_raw.div(nb_athl_total)*100).reset_index().round(1)
nb_athl_raw_pct['Date'] = nb_athl_raw_pct['Date'].dt.strftime('%Y')

px.line(nb_athl_raw_pct, x='Date', y='PCT')


# In[ ]:





# In[ ]:





# In[ ]:




