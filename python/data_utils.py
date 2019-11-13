import re
import pandas as pd
import os
import time


def json_to_tsv(json_file, chunksize=1000):
    """
    Convert a json file to a tsv file
    :param json_file: path to the .json file
    :param chunksize: the size of chunk to convert at each step
    :return: Save a .tsv file in the same folder as the json file
    """

    output_file = re.sub(r'\.json', '.tsv', json_file)
    temp_file = re.sub(r'\.json', '_temp.tsv', json_file)

    json_loader = pd.read_json(json_file, lines=True, chunksize=chunksize)

    for df in json_loader:
        pd.DataFrame(data=None, columns=df.columns).to_csv(output_file, index=False, header=True, sep='\t')
        break

    for index, df in enumerate(json_loader):
        tic = time.time()

        df.to_csv(temp_file, index=False, header=False, sep='\t')
        with open(temp_file, 'r') as f:
            text = f.read()

        with open(output_file, 'a') as f:
            f.write(text)

        toc = time.time()
        print('Iteration {} finished in {:2.4f} ms'.format(index, toc - tic))

    os.remove(temp_file)

