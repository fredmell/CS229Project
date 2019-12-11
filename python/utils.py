import pandas as pd
import numpy as np
import re

def get_skewed_columns(df):
    """ Return columns for which the 0th and 90th quantile are equal.

    """
    user_describe = df.describe(percentiles=[.25, .5, .75, 0.9, 0.95, 0.99])
    columns_to_remove = []
    for i in range(len(user_describe.columns)):
        q0 = user_describe[user_describe.columns[i]]['min']
        q90 = user_describe[user_describe.columns[i]]['90%']
        if q0 == q90:
            columns_to_remove.append(user_describe.columns[i])
    return columns_to_remove

def unnest_dictionary(nested_dico):
    """ Unnest dictionary of attributes.

    """
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

def to_double(element):
    if pd.isna(element):
        return np.nan
    else:
        return np.double(element)

def to_str(element):
    if pd.isna(element):
        return np.nan
    else:
        return str(element)

def find_most_common_value(element_list):

    for element in element_list:
        if not pd.isna(element):
            break

    if pd.isna(element):
        return np.nan

    elif isinstance(element, np.double):
        array = np.array(element_list)
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return np.nan
        else:
            array = array.astype(np.int)
            return np.double(np.bincount(array).argmax())

    elif isinstance(element, str):
        count = Counter(df[col])
        try:
            del count[np.nan]
        except ValueError:
            pass
        if count == dict():
            return np.nan
        else:
            return count.most_common(1)[0][0]

def print_split(df):
    # print('='*80)
    # print('RAW REVIEWS DATASET DESCRIPTION')
    # nb_reviews = len(review_df)
    # pos_reviews = sum(review_df['label'])
    # neg_revews = nb_reviews - pos_reviews

    # print('Total reviews: {:,d}'.format(nb_reviews))
    # print('Positive reviews: {:,d} ({:.2f}%)'.format(pos_reviews, 100*pos_reviews/nb_reviews))
    # print('Negative reviews: {:,d} ({:.2f}%)'.format(neg_revews, 100*neg_revews/nb_reviews))
    # print('='*80)
    
    print('='*80)
    print('REVIEWS DATASET AFTER SPLIT DESCRIPTION')
    nb_reviews = len(df)
    pos_reviews = sum(df['label'])
    neg_revews = nb_reviews - pos_reviews
    train_reviews = len(df[df['dataset'] == 'train'])
    val_reviews = len(df[df['dataset'] == 'val'])
    test_reviews = len(df[df['dataset'] == 'test'])

    print('Positive reviews: {:,d} ({:.2f}%)'.format(pos_reviews, 100*pos_reviews/nb_reviews))
    print('Negative reviews: {:,d} ({:.2f}%)'.format(neg_revews, 100*neg_revews/nb_reviews))
    print()
    print('Train reviews: {:,d} ({:.2f}%)'.format(train_reviews, 100*train_reviews/nb_reviews))
    print('Validation reviews: {:,d} ({:.2f}%)'.format(val_reviews, 100*val_reviews/nb_reviews))
    print('Test reviews: {:,d} ({:.2f}%)'.format(test_reviews, 100*test_reviews/nb_reviews))
    print('='*80)

    print('TRAIN REVIEWS DATASET DESCRIPTION')
    nb_reviews = len(df[df['dataset'] == 'train'])
    pos_reviews = sum(df[df['dataset'] == 'train']['label'])
    neg_revews = nb_reviews - pos_reviews

    print('Total reviews: {:,d}'.format(nb_reviews))
    print('Positive reviews: {:,d} ({:.2f}%)'.format(pos_reviews, 100*pos_reviews/nb_reviews))
    print('Negative reviews: {:,d} ({:.2f}%)'.format(neg_revews, 100*neg_revews/nb_reviews))
    print('='*80)

    print('VALIDATION REVIEWS DATASET DESCRIPTION')
    nb_reviews = len(df[df['dataset'] == 'val'])
    pos_reviews = sum(df[df['dataset'] == 'val']['label'])
    neg_revews = nb_reviews - pos_reviews

    print('Total reviews: {:,d}'.format(nb_reviews))
    print('Positive reviews: {:,d} ({:.2f}%)'.format(pos_reviews, 100*pos_reviews/nb_reviews))
    print('Negative reviews: {:,d} ({:.2f}%)'.format(neg_revews, 100*neg_revews/nb_reviews))
    print('='*80)

    print('TEST REVIEWS DATASET DESCRIPTION')
    nb_reviews = len(df[df['dataset'] == 'test'])
    pos_reviews = sum(df[df['dataset'] == 'test']['label'])
    neg_revews = nb_reviews - pos_reviews

    print('Total reviews: {:,d}'.format(nb_reviews))
    print('Positive reviews: {:,d} ({:.2f}%)'.format(pos_reviews, 100*pos_reviews/nb_reviews))
    print('Negative reviews: {:,d} ({:.2f}%)'.format(neg_revews, 100*neg_revews/nb_reviews))
