import pandas as pd
import time
import utils

from datetime import datetime

class DataLoader:
    """ Utils to load Yelp dataset data files and merge these to one data
    frame, with one row for each review. Each row holds business and user data
    and a label 1.0 for a 5 star rating and 0.0 for ratings of 1 to 4 stars.

    Args:
        path (string) : Path to directory containing yelp data files.
    """
    def __init__(self, path="../data/"):
        self.path = path

    def load_review(self):
        """ Load review data and form a data frame.

        """
        start = time.time()
        print("Loading review ...")

        df = pd.read_json(self.path + "review.json", lines=True)

        df.dropna(inplace=True)

        # Map 5 star rating -> label = 1.0 and 1-4 star ratings -> label = 0.0
        df['label'] = list(map(lambda x: int(int(x) == 5), df['stars'].values))

        # Keep desired columns
        df = df[['review_id', 'business_id', 'user_id', 'label', 'date']]

        t = time.time() - start
        print("Loaded review in {:.1f} seconds".format(t))

        return df


    def load_user(self):
        """ Load user data, count number of friends and add seniority feature.

        Returns:
            df (pd.DataFrame) : Data for each user

        """
        start = time.time()
        print("Loading user ...")

        df = pd.read_json(self.path + "user.json", lines=True)

        df = df[df['review_count'] > 0] # Remove users with no reviews

        # Add seniority feature by mapping the date of user creation to number
        # of days since creation on 12/31/2019.
        date_to_day = lambda d: datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
        df['yelping_since'] = list(map(date_to_day, df['yelping_since']))

        # Count number of friends and add this as a feature
        count_friends = lambda d: len(d.split(','))
        df['nb_friends'] = list(map(count_friends, df['friends']))

        # Remove elite and generic columns
        df.drop(["elite", "name", "yelping_since", "friends"],
                axis=1,
                inplace=True)

        # Get and remove heavily skewed columns which add little information
        columns_to_remove = utils.get_skewed_columns(df)
        df.drop(columns_to_remove, axis=1, inplace=True)

        t = time.time() - start
        print("Loaded user in {:.1f} seconds".format(t))

        return df


    def load_business(self):
        """ Load business data and form a dataframe.

        """
        start = time.time()
        print("Loading business ...")

        df = pd.read_json("../data/business.json", lines=True)

        df = df[df['review_count'] > 0] # Remove businesses with no reviews
        df.dropna(inplace=True)

        fetcher = lambda x: utils.unnest_dictionary(eval(str(x)))
        for key in ["attributes", "hours"]:
            df[key] = list(map(fetcher, df[key].values))

        df = pd.concat([df.drop(['attributes'], axis=1),
                        df['attributes'].apply(pd.Series)],
                       axis=1)

        df = pd.concat([df.drop(['hours'], axis=1),
                        df['hours'].apply(pd.Series)],
                       axis=1)

        t = time.time() - start
        print("Loaded user in {:.1f} seconds".format(t))

        return df

    def merge(self, filename=None):
        """ Merge separate data frames

        """

        # Load data frames
        user     = self.load_user()
        review   = self.load_review()
        business = self.load_business()

        print("Merging ...")

        t0 = time.time()
        df = pd.merge(review, user, on='user_id', how='inner')

        # Free some memory
        del user
        del review

        df = pd.merge(df, business, on='business_id', how='inner')

        t = time.time() - t0
        print("Merged successfully in {:.1f} seconds".format(t))

        # Rename some duplicate columns, remove some unwanted columns
        df.rename(columns={"review_count_x": "user_review_count",
                           "review_count_y": "business_review_count"},
                  inplace=True)

        df.drop(['name', 'address', 'city', 'state'], axis=1, inplace=True)

        if filename is not None:
            df.to_pickle(self.path + filename)

        return df

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.merge()
