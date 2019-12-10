import numpy as np
import pandas as pd
import time
import utils
import warnings

from loaddata import DataLoader
from cat_to_num import *

class Preprocess:
    """ Preprocesses data and performs data split.

    """
    def __init__(self):
        pass

    def rm_weekdays(self, df):
        """ Remove weekday columns in-place.

        Args:
            df (pd.DataFrame) : Data with columns "Monday" through "Sunday"

        Returns:
            None (NoneType)

        """
        num_predictors = len(df.columns)

        df.drop(labels=["Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday"],
                axis=1,
                inplace=True)

        # Check that 7 columns were removed
        assert num_predictors == len(df.columns) + 7

    def categorical_to_numerical(self, df):
        """ Convert categorical variables to numerical variables. Also convert
        all other variables to numerical if possible.

        """

        for j, column in enumerate(df):
            if column in cat_to_num_mapping:
                df[column] = df[column].map(lambda x: cat_to_num_mapping[column][x], na_action="ignore")
            else:
                df[column] = pd.to_numeric(df[column], errors="ignore")

            print('{:03d} {:35s}'.format(j, column), type(df[column][0]))

        return df

    def fill_na(self, df):
        """ Fill na values with mode of column in-place

        """
        # Fill missing values with mode
        ignored_columns = ['review_id', 'business_id', 'user_id', 'postal_code', 'categories', 'date']
        for j, column in enumerate(df):
            if column not in ignored_columns:
                try:
                    mode = df[column].mode().iloc[0]
                except IndexError:
                    print("Warning: No mode, imputed with NaN.")
                    mode = np.nan
                df[column].fillna(value=mode, inplace=True)
                print('{:03d} {:35s}'.format(j, column), mode)
