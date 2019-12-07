"""
Fill na with most common of the whole column
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
file_na = '/home/nicolasbievre/yelp_data_no_na.pkl'


df = pd.read_pickle(file)

categories = list(set(df['categories'].values))
n = len(categories)

for i in tqdm(range(len(df.columns))):
    col = df.columns[i]

    if not col in {'review_id': 0, 'business_id': 0, 'user_id': 0, 'postal_code': 0}:

        df_col = df[col].values

        na = sum(pd.isna(df_col))
        if na > 0:
            most_commom_term = find_most_common_value(df_col)

            if not pd.isna(most_commom_term):
                    df.loc[(pd.isna(df_col)), col] = most_commom_term
    if i % 35 == 0 and i > 0:
        df.to_pickle(file_na)

df.to_pickle(file_na)
