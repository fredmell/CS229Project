import numpy as np
import pandas as pd
import utils

from tqdm import tqdm
from loaddata import DataLoader
from cat_to_num import *

class Preprocess:
    """ Preprocesses data and performs data split. See individual method
    docstrings for details. Each method modifies the dataframe attribute
    in-place.

    """
    def __init__(self, df):
        self.df = df

    def rm_weekdays(self):
        """ Remove weekday columns in-place.

        Args:
            df (pd.DataFrame) : Data with columns "Monday" through "Sunday"

        Returns:
            None (NoneType)

        """
        num_predictors = len(self.df.columns)

        self.df.drop(labels=["Monday",
                             "Tuesday",
                             "Wednesday",
                             "Thursday",
                             "Friday",
                             "Saturday",
                             "Sunday"],
                     axis=1,
                     inplace=True)

        # Check that 7 columns were removed
        assert num_predictors == len(self.df.columns) + 7

    def categorical_to_numerical(self):
        """ Convert categorical variables to numerical variables. Also convert
        all other variables to numerical if possible.

        """

        for j, column in enumerate(self.df):
            if column in cat_to_num_mapping:
                mapping = lambda x: cat_to_num_mapping[column][x]
                self.df[column] = self.df[column].map(mapping,
                                                      na_action="ignore")
            else:
                self.df[column] = pd.to_numeric(self.df[column],
                                                errors="ignore")

            print('{:03d} {:35s}'.format(j, column), type(self.df[column][0]))

        self.df['date'] = pd.to_datetime(self.df['date'])

    def fill_na(self):
        """ Fill na values with mode of column in-place.

        """
        # Fill missing values with mode
        ignored_columns = ['review_id',
                           'business_id',
                           'user_id',
                           'postal_code',
                           'categories',
                           'date']

        for j, column in enumerate(self.df):
            if column not in ignored_columns:
                try:
                    mode = self.df[column].mode().iloc[0]
                # Used for testing on subsets of data, should not occur
                except IndexError:
                    print("Warning: No mode, imputed with NaN.")
                    mode = np.nan

                self.df[column].fillna(value=mode, inplace=True)
                print('{:03d} {:35s}'.format(j, column), mode)

    def count_categories(self):
        """ Helper function to count the number of appearences of categories.

        Returns:
            sorted_vals (list) : Each entry a (category, count) pair, sorted by
                                 counts.

        """
        business_categories = self.df['categories']

        b_category_counts = {}

        # Iterate over reviews
        for row in range(business_categories.shape[0]):
            # Skip nones
            if business_categories[row] is None:
                continue

            # Cut whitespace
            categories = set(business_categories[row].strip().split(","))

            # Increment dictionary entries
            for cat in categories:
                if cat[0] == " ":
                    cat = cat[1:]
                if cat in b_category_counts.keys():
                    b_category_counts[cat] += 1
                else:
                    b_category_counts[cat] = 1

        # Sort values and print
        sorted_vals = []
        for key in b_category_counts.keys():
            sorted_vals.append((key, b_category_counts[key]))
        sorted_vals = sorted(sorted_vals, key=lambda x: x[1], reverse = True)

        return sorted_vals

    def unravel_categories(self, obs_thresh=100000):
        """ Unravel categories with number of observations greater than
        obs_thresh in-place. Adds a new column for each of these categories,
        and one-hot encodes the precense of a category for a given business.

        Args:
            df (pd.DataFrame) : Dataset with column 'categories' to be unraveled
            obs_thresh (int)  : Threshold of number of observations of category

        Returns:
            df : Data frame with unraveled category columns

        """

        counts = self.count_categories()

        new_columns = []
        for category, num in tqdm(counts):
            if num >= obs_thresh:
                new_columns.append(category)
                self.df[category] = 1
                self.df[category].where(self.df.categories.str.contains(category),
                                        0,
                                        inplace=True)

    def sort_by_date(self):
        self.df.sort_values(by='date', axis=0, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def split_data(self, splits=[0.8230, 0.9115]):
        """ Split data into a train, validation and test set. The splits
        indicate what quantiles to split at, the default is a 2004-2017
        train set, a half 2018 validation set and the remaining half of 2018
        reviews for the test set. Modifies the data in-place. A column
        dataset is added indicating which dataset an observation belongs to.

        Args:
            splits (iterable) : Length 2 iterable with quantiles at which to
                                split the data.

        """
        quantiles = df.date.quantile(splits, interpolation='nearest')

        self.df['dataset'] = "train"

        self.df['dataset'].where(self.df.date < quantiles[splits[0]],
                                 other="val",
                                 inplace=True)

        self.df['dataset'].where(self.df.date < quantiles[splits[1]],
                                 other="test",
                                 inplace=True)

    def print_split(self):
        utils.print_split(df)

    def dump(self, filename="../data/yelp_df.pkl"):
        """ Dump current data frame to file named filename.

        """
        self.df.to_pickle(filename)
