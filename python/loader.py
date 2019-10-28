import json
import os
import pandas as pd


class DataLoader:
    """ Load and do very basic preprocessing on data """
    def __init__(self, path=None):
        pass

    def load_review(self):
        

    def load_user(self):
        """ This .json file is 2.3 Gigs. Need to use pd.read_json in multiple
        batches, or manually read line by line.
        TODO: Find appropriate way to load this file and extract parts of data.

        """

    def load_business(self, min_reviews=50, rm_cols=True, save_csv=False):
        """ Business file.

        Args:
            min_reviews (int) : Only save businesses with min_reviews or more
            rm_cols (bool)    : If True, remove unwanted? columns
            save_csv (bool)   : If True, save filtered df to csv

        Returns:
            df (pd.DataFrame) : Filtered business data

        """
        df = pd.read_json("../data/business.json", lines=True)

        # Remove unwanted? columns. Can change this, e.g. pass list of
        # columns to remove
        if rm_cols:
            df.drop(['address',
                     'postal_code',
                     'city',
                     'state'],
                     axis=1, inplace=True)

        # Remove businesses with few reviews
        df = df[df.review_count >= min_reviews]

        if save_csv:
            df.to_csv("../data/business.csv")

        return df

if __name__ == "__main__":
    loader = DataLoader()
    business = loader.load_business()
    print(business.head())
