import pandas as pd
import time
import pickle
from data_utils import unnest_dictionary as unnest_dict

start = time.time()

print("Reading df")
df = pd.read_pickle("../data/yelp_df.pkl")
print("Read df in {}s".format(time.time() - start))
start = time.time()

print("Unnesting dictionaries")
# Add empty dicts
attributes = df['attributes_b'].apply(lambda x: x if x is not None else {})

# Unnest dicts
attributes = attributes.apply(unnest_dict)
print("Unnested attr in {}s".format(time.time() - start))
start = time.time()

print("Counting observations")
# Count shit
unique_attributes = set()
#observations = dict()
num_occurences = dict()

# Count and store values
for d in attributes:
    for attr, val in d.items():

        if attr not in unique_attributes:
            unique_attributes.add(attr)
            #observations[attr] = [val]
            num_occurences[attr] = 1
        else:
            num_occurences[attr] += 1
            #observations[attr].append(val)
print("Counted attr in {}s".format(time.time() - start))
start = time.time()
# Print first 50 in sorted order
for key in reversed(sorted(num_occurences, key=num_occurences.get)):
    # print(key)
    print("{} : {}".format(num_occurences[key], key))

with open("attribute_count.pkl", "wb") as f:
    pickle.dump(num_occurences, f, protocol=pickle.HIGHEST_PROTOCOL)
