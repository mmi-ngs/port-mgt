import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


os.chdir('.\Projects\Peer_Comparison')
os.getcwd()

# Read data
peer_daa = pd.read_excel('MySuper Peer Comparison_20210413.xlsx',
                         sheet_name='clean')
sectors = [*peer_daa.columns[1:-1]]
funds = [*peer_daa['Fund Name']]

# Put it into long form
list_output = []
for col in sectors:
    df_i = peer_daa.loc[:, ['Fund Name', col]]
    df_i['Sector'] = df_i.columns[-1]
    df_i.columns = ['Fund Name', 'Actual Weight', 'Sector']
    list_output.append(df_i)

peer_daa_long = pd.concat(list_output, axis=0)

# Catplot
g = sns.catplot(x='Fund Name', y='Actual Weight', col='Sector', col_wrap=4,
                kind='bar', data=peer_daa_long, palette=sns.color_palette())
(g.set_titles("{col_name} {col_var}")
  .set_axis_labels("", "Weight")
  .set_xticklabels(funds, rotation=90))
g.fig.suptitle('SuperRatings Asset Allocation Peer Comparison')