import pandas as pd

from loaddata import DataLoader
from preprocess import Preprocess

def main():
    loader = DataLoader(path="../data/")
    df = loader.merge()

    preprocessor = Preprocess(df)
    preprocessor.rm_weekdays()
    preprocessor.categorical_to_numerical()
    preprocessor.fill_na()
    preprocessor.unravel_categories()
    preprocessor.sort_by_date()
    preprocessor.dump("../data/yelp_df.pkl")

if __name__ == "__main__":
    main()
