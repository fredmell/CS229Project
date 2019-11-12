import json
import os
import pandas as pd
import time


class DataLoader:
    """ Load and do very basic preprocessing on data.

    Put the data files in a directory named data, in the repository.

    To make the combined dataset, first run the to_csv method to strip
    the reviews of their text and then write this resulting dataframe to csv.

    Then run make_combined to make a combined dataframe with review, user and
    business data for the first n reviews. 
    """
    def __init__(self, path="../data/"):
        self.path = path

    def to_csv(self):
        df = pd.read_json(self.path + "review.json", lines=True)
        df.drop(['text'], axis=1, inplace=True)
        df.to_csv("../data/review.csv")

    def load_review(self, n=None):
        """

        Args:
            n         (int) : Number of reviews to read. Defaults to None, for
                              which all lines are read.

        """
        df = pd.read_csv(self.path + "review.csv", nrows=n)

        return df


    def load_user(self):
        """

        """
        df = pd.read_json(self.path + "user.json", lines=True)

        return df


    def load_business(self, min_reviews=1, rm_cols=False, save_csv=False):
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

    def make_combined(self, n=None, to_csv=False):
        review = self.load_review(n)
        review.drop(['Unnamed: 0'], axis=1, inplace=True)

        print("Loading business")
        business = self.load_business()
        print("Loading user")
        user = self.load_user()

        print("Making merged dataframe")

        t0 = time.time()
        merged = pd.merge(review, business, on="business_id")
        merged = pd.merge(merged, user, on="user_id")
        t = time.time() - t0

        print(merged)
        print(merged.columns)

        merged.to_csv("../data/combined{}.csv".format(n))


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.make_combined(100)
