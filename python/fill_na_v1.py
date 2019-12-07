"""
Fill na with most common of the same group category column
"""



import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime
import re
from collections import Counter
from statistics import median
from tqdm import tqdm


def find_most_common_value(element_list):

    for element in element_list:
        if not pd.isna(element):
            break

    if pd.isna(element):
        return np.nan

    elif isinstance(element, np.double):
        array = np.array(element_list)
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return np.nan
        else:
            array = array.astype(np.int)
            return np.double(np.bincount(array).argmax())

    elif isinstance(element, str):
        count = Counter(df[col])
        try:
            del count[np.nan]
        except ValueError:
            pass
        if count == dict():
            return np.nan
        else:
            return count.most_common(1)[0][0]


file = '/home/nicolasbievre/yelp_data.pkl'
file_na = '/home/nicolasbievre/yelp_data_no_na_group.pkl'


df = pd.read_pickle(file)

categories = list(set(df['group_categories'].values))

for i in tqdm(range(len(categories))):
    category = categories[i]
    cat_idx = df['group_categories'] == category
    sub_df = df[cat_idx]

    for j in range(len(sub_df.columns)):
        col = sub_df.columns[j]

        sub_df_col = sub_df[col]
        sub_df_col_na = pd.isna(sub_df_col)
        na = sum(sub_df_col_na)
        if na > 0:
            most_common_value = find_most_common_value(sub_df_col)

            if not pd.isna(most_common_value):
                df.loc[(sub_df_col_na) & (cat_idx), col] = most_common_value

    if i % 550 == 0 and i > 0:
        df.to_pickle(file_na)


df.to_pickle(file_na)
