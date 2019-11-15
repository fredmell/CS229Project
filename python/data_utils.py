import re
import pandas as pd
import os
import time
import numpy as np


def unnest_dictionary(nested_dico):
    unnested_dico = nested_dico.copy()
    finished = False
    while not finished:
        finished = True

        keys = list(unnested_dico.keys())

        for key in keys:

            attributes = str(unnested_dico[key])
            if re.match(r'{.*:.*}', attributes):

                # Sub dico with the keys
                key_dico = eval(attributes)
                sub_dico = {'{}_{}'.format(key, elt): str(key_dico[elt]) for elt in key_dico}

                # Update output dictionary
                unnested_dico.pop(key, None)
                unnested_dico = {**unnested_dico, **sub_dico}

                # Since we found at least one value which one a dictionary we set finished
                # to false to do another loop and the check if the value that was unnested
                # could be even more unnested
                finished = False

    unnested_dico = {key: re.sub(r"[a-z]'(.*)'", r'\1', str(unnested_dico[key])) for key in unnested_dico}
    unnested_dico = {key: re.sub(r"'", '', unnested_dico[key]) for key in unnested_dico}
    return unnested_dico



def json_to_tsv(json_file, output_file=None, chunksize=1000, pre_process=None):
    """
    Convert a json file to a tsv file
    :param json_file: path to the .json file
    :param output_file: name of the output file
    :param chunksize: the size of chunk to convert at each step
    :param pre_process: function to pre process a data frame

    :return: Save a .tsv file in the same folder as the json file
    """

    # Name of the output tsv file
    if output_file is None:
        output_file = re.sub(r'\.json', '.tsv', json_file)

    # Name of the temp file used in the conversion process
    temp_file = re.sub(r'\.tsv', '_temp.tsv', output_file)

    # Json Loader
    json_loader = pd.read_json(json_file, lines=True, chunksize=chunksize)

    # Create an empty .tsv file with the same header as the original .json file
    for df in json_loader:

        # Pre process data
        if pre_process:
            df = pre_process(df)

        pd.DataFrame(data=None, columns=df.columns).to_csv(output_file, index=False, header=True, sep='\t')
        break

    # Convert each chunk of .json file to .tsv
    for index, df in enumerate(json_loader):

        # Start Clock
        tic = time.time()

        # Pre process data
        if pre_process:
            df = pre_process(df)

        # Convert the chunk to a .tsv file
        df.to_csv(temp_file, index=False, header=False, sep='\t')
        with open(temp_file, 'r') as f:
            text = f.read()

        # Load the chunk as a text file to update the output .tsv file
        # without loading it
        with open(output_file, 'a') as f:
            f.write(text)

        # Stop clock and print status
        toc = time.time()
        print('Iteration {} finished in {:2.4f} ms'.format(index, toc - tic))

    # Delete temp file
    os.remove(temp_file)











"""

def pre_process_review(raw_df):
    df = df[['user_id', 'business_id', 'stars']].copy()
    df['like'] = np.array(np.array(df['stars']) == 5, dtype=np.double)
    df.drop(['stars'], axis=1, inplace=True)

    return df


def pre_process_business(raw_df):
    df = raw_df.copy()






def pre_process_user(df):
    df.drop(['text', 'date'], inplace=True)

#json_to_tsv('../data/review.json', chunksize=1000)

#review = pd.read_csv('../data/review.tsv', sep='\t')

#print(review.head())


#print(len(review))


def friends_split(df):

    data = {'user_id': [], 'friend_id': []}

    for _, row in df.iterrows():
        user_id = row['user_id']
        friends = re.split(r', ?', str(row['friends']))
        if friends != []:
            for friend in friends:
                data['user_id'].append(user_id)
                data['friend_id'].append(friend)

    return pd.DataFrame.from_dict(data)


#json_to_tsv('../data/user.json', output_file='../data/friends.tsv', chunksize=10000, pre_process=friends_split)



#friends = pd.read_csv('../data/friends.tsv', sep='\t')
#print(friends.head())



dico1 = {"RestaurantsReservations":"True","GoodForMeal":"{'dessert': False, 'latenight': False, 'lunch': True, 'dinner': True, 'brunch': False, 'breakfast': False}","BusinessParking":"{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}","Caters":"True","NoiseLevel":"u'loud'","RestaurantsTableService":"True","RestaurantsTakeOut":"True","RestaurantsPriceRange2":"2","OutdoorSeating":"False","BikeParking":"False","Ambience":"{'romantic': False, 'intimate': False, 'classy': False, 'hipster': False, 'divey': False, 'touristy': False, 'trendy': False, 'upscale': False, 'casual': True}","HasTV":"False","WiFi":"u'no'","GoodForKids":"True","Alcohol":"u'full_bar'","RestaurantsAttire":"u'casual'","RestaurantsGoodForGroups":"True","RestaurantsDelivery":"False"}
dico = {"business_id":"gnKjwL_1w79qoiV3IC_xQQ","name":"Musashi Japanese Restaurant","address":"10110 Johnston Rd, Ste 15","city":"Charlotte","state":"NC","postal_code":"28210","latitude":35.092564,"longitude":-80.859132,"stars":4.0,"review_count":170,"is_open":1,"attributes":{"GoodForKids":"True","NoiseLevel":"u'average'","RestaurantsDelivery":"False","GoodForMeal":"{'dessert': False, 'latenight': False, 'lunch': True, 'dinner': True, 'brunch': False, 'breakfast': False}","Alcohol":"u'beer_and_wine'","Caters":"False","WiFi":"u'no'","RestaurantsTakeOut":"True","BusinessAcceptsCreditCards":"True","Ambience":"{'romantic': False, 'intimate': False, 'touristy': False, 'hipster': False, 'divey': False, 'classy': False, 'trendy': False, 'upscale': False, 'casual': True}","BusinessParking":"{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}","RestaurantsTableService":"True","RestaurantsGoodForGroups":"True","OutdoorSeating":"False","HasTV":"True","BikeParking":"True","RestaurantsReservations":"True","RestaurantsPriceRange2":"2","RestaurantsAttire":"'casual'"},"categories":"Sushi Bars, Restaurants, Japanese","hours":{"Monday":"17:30-21:30","Wednesday":"17:30-21:30","Thursday":"17:30-21:30","Friday":"17:30-22:0","Saturday":"17:30-22:0","Sunday":"17:30-21:0"}}

test = {"a": "{'b': 1, 'c': {'d': 2, 'e': 3}}", "f": 4}
rst = unnest_dictionary(dico)
print()
for key in rst:
    print(key)
    print(rst[key])
    print()

# json_to_tsv('../data/review.json', pre_process=pre_process_review, chunksize=1000)
# json_to_tsv('../data/business_sample.json', chunksize=1000)
# json_to_tsv('../data/user_sample.json', chunksize=1000)





def create_ratings_dataset(review_file, business_file, user_file, output_file,
                           review_features=None, business_features=None, user_features=None):

    # Load reviews data
    review = pd.read_csv(review_file, sep='/t')
    if review_features:
        review = review[review_features]

    # Load business data
    business = pd.read_csv(business_file, sep='/t')
    if business_features:
        business = business[business_features]

    # Load user data
    user = pd.read_csv(user_file, sep='/t')
    if user_features:
        user = user[user_features]

    # Merge dataset
    review = review.merge(user, how='left', on='user_id')
    review = review.merge(business, how='left', on='business_id')

    # Save results
    review.to_csv(output_file, index=False, header=False, sep='\t')



pd.concat([df.drop([''], axis=1), df['b'].apply(pd.Series)], axis=1)


create_ratings_dataset(review_file='../data/review_sample.tsv',
                       business_file='../data/business_sample.tsv',
                       user_file='../data/user_sample.tsv',
                       output_file='../data/ratings.tsv',
                       review_features=['user_id', 'business_id', 'stars'],
                       business_features=['user_id', 'business_id', 'stars'],
                       user_features=['user_id', 'review_count', 'fans'],)


# json_to_tsv('../data/review.json', chunksize=1000)
"""