import pandas as pd

from loaddata import DataLoader
from preprocess import Preprocess

def main():
    # Load and merge review, business and user data
    loader = DataLoader(path="../data/")
    df = loader.merge()

    preprocessor = Preprocess(df)

    # Remove weekday columns
    preprocessor.rm_weekdays()

    # Convert categorical variables to one-hot encoded variables and convert
    # strings to numerical types if possible
    preprocessor.categorical_to_numerical()

    # Fill NA observations with the mode of the given feature
    preprocessor.fill_na()

    # Unravel and one-hot encode business categories
    preprocessor.unravel_categories()

    # Sort observations by date and reindex with this ordering
    preprocessor.sort_by_date()

    # Form temporal train-val-test split
    preprocessor.split_data()

    # Write the final dataframe to a pickle file
    preprocessor.dump("../data/yelp_df.pkl")

if __name__ == "__main__":
    main()
