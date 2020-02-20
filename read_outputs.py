import pandas as pd
import numpy as np
import json
import os
import copy

#Read in file
folder = 'convolution_output'
data_file = input('csv file for data:')
file_path = os.path.join(folder, data_file)
df = pd.read_csv(file_path)

#process so that it is usable
def convert_to_list(elem):
    #remove brackets from string
    elem = elem.replace('[', '')
    elem = elem.replace(']', '')
    elem = elem.replace('(', '')
    elem = elem.replace(')', '')
    elem = elem.replace('{', '')
    elem = elem.replace('}', '')
    #split at ', ' and make each element a float
    tmp_list = list(elem.split(', '))
    num_list = list(map(lambda s: float(s), tmp_list))
    return num_list

for c in df.columns:
    if df[c].dtypes == 'O':
        df[c]= df[c].apply(convert_to_list)


#compare number to list
#note, even though variable name is vector, it is a list
def find_percentile(number, vector):
    #copy vector so python doesn't screw it up
    working_copy = vector.copy()
    #add the number we are intersted in to the end
    working_copy.append(number)
    print(working_copy)
    #sort
    working_copy.sort()
    #find number in sorted list
    position = working_copy.index(number)
    #calulate percentile
    percentile = position / (len(working_copy)-1)
    return percentile


def create_percentile_column(num_name, list_name):
    row = list(df.index)
    percentile_list = []
    for i in row:
        percentile = find_percentile(df.ix[i,num_name], df.ix[i,list_name])
        percentile_list.append(percentile)
    return percentile_list

df["percentile dd distance from mean"] = create_percentile_column('original dd distance from mean', 'sample distances from mean dd vec')
df["percentile ev distance from mean"] = create_percentile_column('original ev distance from mean', 'sample distances from mean ev vec')
df['percentile triangle count'] = create_percentile_column('original triangles', 'sample triangle counts')

print('now what')

