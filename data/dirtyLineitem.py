import pandas as pd
import numpy as np
from math import trunc
from datetime import datetime
import os

li_cols = [
    'L_ORDERKEY',
    'L_PARTKEY',
    'L_SUPPKEY',
    'L_LINENUMBER',
    'L_QUANTITY',
    'L_EXTENDEDPRICE',
    'L_DISCOUNT',
    'L_TAX',
    'L_RETURNFLAG',
    'L_LINESTATUS',
    'L_SHIPDATE',
    'L_COMMITDATE',
    'L_RECEIPTDATE',
    'L_SHIPINSTRUCT',
    'L_SHIPMODE',
    'L_COMMENT',
]

df_original = pd.read_table('lineitem.tbl', comment='#', sep='|', delim_whitespace=False, names =li_cols, header=None, index_col=False)
df = df_original.copy()

start = pd.DataFrame({'A': ['1990-01-01'] })
start['A'] = pd.to_datetime(start['A']).astype('int')
startV = start.A.max()
#print (startV)

# clean data -- remove dates
df['L_SHIPDATE'] = pd.to_datetime(df['L_SHIPDATE']).astype('int') - startV
df['L_SHIPDATE'] /= 10000000000
df["L_SHIPDATE"] = df['L_SHIPDATE'].astype('int')

df['L_COMMITDATE'] = pd.to_datetime(df['L_COMMITDATE']).astype('int') - startV
df['L_COMMITDATE'] /= 10000000000
df["L_COMMITDATE"] = df['L_COMMITDATE'].astype('int')

df['L_RECEIPTDATE'] = pd.to_datetime(df['L_RECEIPTDATE']).astype('int') - startV
df['L_RECEIPTDATE'] /= 10000000000
df["L_RECEIPTDATE"] = df['L_RECEIPTDATE'].astype('int')

# clean data -- remove strings
rf = df['L_RETURNFLAG'].unique()
rf_dict = dict(zip(rf, range(len(rf))))
df['L_RETURNFLAG']= df['L_RETURNFLAG'].map(rf_dict)

ls = df['L_LINESTATUS'].unique()
ls_dict = dict(zip(ls, range(len(ls))))
df['L_LINESTATUS']= df['L_LINESTATUS'].map(ls_dict)

lss = df['L_SHIPINSTRUCT'].unique()
lss_dict = dict(zip(lss, range(len(lss))))
df['L_SHIPINSTRUCT']= df['L_SHIPINSTRUCT'].map(lss_dict)

lsm = df['L_SHIPMODE'].unique()
lsm_dict = dict(zip(lsm, range(len(lsm))))
df['L_SHIPMODE']= df['L_SHIPMODE'].map(lsm_dict)

lc = df['L_COMMENT'].unique()
lc_dict = dict(zip(lc, range(len(lc))))
df['L_COMMENT']= df['L_COMMENT'].map(lc_dict)

# clean data -- remove floats
df["L_EXTENDEDPRICE"] = df["L_EXTENDEDPRICE"].round()
df["L_EXTENDEDPRICE"] = df["L_EXTENDEDPRICE"].astype('Int64')

df["L_DISCOUNT"] *= 100
df["L_DISCOUNT"] = df["L_DISCOUNT"].astype('int')

df["L_TAX"] *= 100
df["L_TAX"] = df["L_TAX"].astype('int')

outputDir = "simplified/"
outputDir2 = "original/"

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

if not os.path.exists(outputDir2):
    os.makedirs(outputDir2)

df.to_csv(outputDir + "lineitem0.csv", index=False)
df_original.to_csv(outputDir2 + "lineitem0.csv", index=False)

# dataset with only one single missing value
df_copy = df.copy()
df_copy.at[2,'L_EXTENDEDPRICE'] = None
df_copy.to_csv(outputDir + "lineitem001.csv", index=False)
df_original_copy = df_original.copy()
df_original_copy.at[2,'L_EXTENDEDPRICE'] = None
df_original_copy.to_csv(outputDir2 + "lineitem001.csv", index=False)

# dataset with 10% to 90% missing values
for i in [0.1, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    num_missing = round(i*len(df.index))
    missing = np.concatenate([np.repeat(True, num_missing), np.repeat(False, len(df.index) - num_missing - 1)])
    np.random.shuffle(missing)
    missing = np.concatenate([np.array([False]), missing]) # no missing value on the first row (unsupported by BOSS)

    df_copy = df.copy()
    df_original_copy = df_original.copy()
    for index in range(len(missing)):
        if missing[index]:
            df_copy.at[index,'L_EXTENDEDPRICE'] = None
            df_original_copy.at[index,'L_EXTENDEDPRICE'] = None

    filepath = outputDir + "lineitem" + str(trunc(i*100)) + ".csv"
    filepath2 = outputDir2 + "lineitem" + str(trunc(i*100)) + ".csv"
    df_copy.to_csv(filepath, index=False)
    df_original_copy.to_csv(filepath2, index=False)

    print(str(i*100) + "% - " + str(num_missing) + " missing values")
