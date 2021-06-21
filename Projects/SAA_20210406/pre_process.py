import pandas as pd
import os


os.chdir(r'.\Projects\SAA_20210406')
os.getcwd()

xls = pd.ExcelFile('ngs_saa.xlsx')
main = pd.read_excel(xls, 'main_202106', index_col=0)
saa = pd.read_excel(xls, 'saa_202106', index_col=0)
# constraints = pd.read_excel(xls, 'constraints')

# Get NGS assets (any option with positive saa weight)
assets_ngs = [*saa.loc[saa.sum(axis=1) > 0].index]
print(len(assets_ngs))

# Get clean versions
saa.loc[assets_ngs].to_clipboard()

main.loc[assets_ngs, ['Before_Tax_Returns',
                      'Before_Tax_Risk_Normal']].to_clipboard()
main.loc[assets_ngs, assets_ngs].to_clipboard()
