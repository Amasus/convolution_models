import pandas as pd
import numpy as np
import json
import os
import copy

#Read in file
folder = 'convolution_output'
#data_file = input('csv file for data:')
data_file = 'mouse_retina_lattice.csv'
file_path = os.path.join(folder, data_file)
df = pd.read_csv(file_path)

#process so that it is usable
#def convert_to_list(elem):
#    #remove brackets from string
#    elem = elem.replace('[', '')
#    elem = elem.replace(']', '')
#    elem = elem.replace('(', '')
#    elem = elem.replace(')', '')
#    elem = elem.replace('{', '')
#    elem = elem.replace('}', '')
#    elem = elem.replace('\n', '')
#    while '  ' in elem:
#        elem = elem.replace('  ', ' ')
#    #split at ', ' and make each element a float
#    if ',' in elem:
#        tmp_list = list(elem.split(', '))
#    else:
#        tmp_list = list(elem.split(' '))
#    num_list = list(map(lambda s: float(s), tmp_list))
#    return num_list

for column in df.columns:
    if isinstance(df[column].iloc[0], str):
        try:
            df[column] = df[column].apply(lambda mat_str: np.array(json.loads(mat_str)).tolist())
        except json.JSONDecodeError:
            # Clearly, despite being a string, that data wasn't meant to be interpretted as being JSON
            pass


#for c in df.columns:
#    if isinstance(df[c].iloc[1], str):
#        print(c)
#        df[c]= df[c].apply(convert_to_list)


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
        percentile = find_percentile(df[num_name].iloc[i], df[list_name].iloc[i])
        percentile_list.append(percentile)
    return percentile_list

min_error = df['error'].min()
min_row = df.loc[df['error'] == min_error]
percentile= find_percentile(min_row['original dd distance from mean'].item(), min_row['sample distances from mean dd vec'].item()[0])
print('original distance', min_row['original dd distance from mean'].item())
print('sample distances', min_row['sample distances from mean dd vec'].item()[0])

df["percentile dd distance from mean"] = create_percentile_column('original dd distance from mean', 'sample distances from mean dd vec')
df["percentile ev distance from mean"] = create_percentile_column('original ev distance from mean', 'sample distances from mean ev vec')
df['percentile triangle count'] = create_percentile_column('original triangles', 'sample triangle counts')

print('now what')

