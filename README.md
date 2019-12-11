# CS229 Project - Yelp Dataset

The contributors to this project are Pranav Bhardwaj, Nicolas Bievre and Frederik J. Mellbye.

## Dataset
The Yelp Open Dataset is available [here](https://www.yelp.com/dataset). To form the dataframe, download the Yelp
Open Dataset and move files `review.json`, `user.json` and `business.json` to directory [data](data). Then, in directory [python](python), run
```
python3 make_dataframe.py
```
This should take while to run. Once complete, the dataframe is stored as a pickle in `data/yelp_df.pkl`.
