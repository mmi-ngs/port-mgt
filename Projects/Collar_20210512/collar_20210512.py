import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


os.chdir(r'.\Projects\Collar_20210512')
os.getcwd()

xls = pd.ExcelFile('xjo_clean_20210512.xlsx')
jul = pd.read_excel(xls, 'Jul', index_col=0)
aug = pd.read_excel(xls, 'Aug', index_col=0)
sep = pd.read_excel(xls, 'Sep', index_col=0)
xjo = pd.read_excel(xls, 'clean', index_col=0)

# Add option type
xjo.reset_index(inplace=True)
mask_call = xjo.ID.str.startswith('C')
xjo.loc[mask_call, 'Type'] = 'Call'
xjo.loc[~mask_call, 'Type'] = 'Put'

# Use OTM options only
xjo['Moneyness'] = round(xjo['Strike/Stock'], 2) * 100
xjo['Moneyness'] = xjo['Moneyness'].astype('int32')

mask_otm = ((xjo['Strike/Stock'] >= 1) & (xjo.Type == 'Call')) | \
           ((xjo['Strike/Stock'] <= 1) & (xjo.Type == 'Put'))
xjo_otm = xjo.loc[mask_otm]
xjo_otm = xjo_otm.loc[xjo_otm['Imp. Vol.'] != 0, :]

'''--------Vol Surface Plot-------'''
vol_surf = sns.catplot(x='Strike/Stock', y='Imp. Vol.', data=xjo_otm,
                       hue='Expiration Date')
vol_surf.set_xticklabels(rotation=30)
vol_surf.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
vol_surf.xaxis.set_major_formatter(ticker.ScalarFormatter)

