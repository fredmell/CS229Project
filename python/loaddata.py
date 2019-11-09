import json
import os
import pandas as pd


class DataLoader:
    """ Load and do very basic preprocessing on data """
    def __init__(self, path="../data/"):
        self.path = path

    def load_review(self, n=None, chunksize=1000000):
        """ This .json file is 5 Gigs. Need to use pd.read_json in multiple
        batches, or manually read line by line.
        TODO: Find appropriate way to load this file and extract parts of data
        without running out of RAM.

        Args:
            n         (int) : Number of reviews to read. Defaults to None, for
                              which all lines are read.
            chunksize (int) : Number of json lines to read to RAM at a time.
                              Defaults to a million. There are 66 million lines in the file.

        """
        pass

    def load_user(self, chunksize=500000):
        """ This .json file is 2.3 Gigs. Need to use pd.read_json in multiple
        batches, or manually read line by line.
        TODO: Find appropriate way to load this file and extract parts of data,
        without running out of RAM.

        It seems we can use kwarg chunksize, read_json then returns an iterator,
        which loads chunksize lines to RAM at a time.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html
        https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json

        Perhaps not needed for user data, but definately needed for review data.

        """
        reader = pd.read_json(self.path + "user.json",
                              lines=True,
                              chunksize=chunksize)

        for chunk in reader:
            print(chunk.memory_usage(deep=True).sum()/1e9, "GB")


    def load_business(self, min_reviews=50, rm_cols=True, save_csv=False, chunksize=None):
        """ Business file.

        Args:
            min_reviews (int) : Only save businesses with min_reviews or more
            rm_cols (bool)    : If True, remove unwanted? columns
            save_csv (bool)   : If True, save filtered df to csv

        Returns:
            df (pd.DataFrame) : Filtered business data

        """
        df = pd.read_json("../data/business.json", lines=True, chunksize=chunksize)

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
    user = loader.load_user()
