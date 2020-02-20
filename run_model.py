import os
import json
import argparse
import numpy as np

###personal modules###
from file_conversion import convert
from convolution import convolUtion_metrics_step, convolution_metrics_lattice
import write_outputs as output


def main():
    #read in parameters from json
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs')
    arguments = parser.parse_args()
    with open(arguments.inputs) as f:
        inputs = json.load(f)
    #collect user inputs
    data_path = inputs['path']
    print(data_path)
    step_size_multiplier = inputs['step_size']
    sample_size = inputs['sample_size']
    print(step_size_multiplier, sample_size)
    output_path = inputs['output_dir']
    critter = inputs['critter_name']+'.csv'
    critter_lattice = inputs['critter_name']+'_lattice.csv'

    #convert the data
    adjacency = convert(data_path)

    #run step convolution model
    #summary = convolUtion_metrics_step(adjacency, step_size_multiplier, sample_size)

    #run the lattice convolution model
    summary, lattice_df = convolution_metrics_lattice(adjacency, step_size_multiplier, sample_size)

    #path to write to
    output_path = os.path.join(output_path)

    # Make path if it doesnt exist
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    #file names
    critter_file = os.path.join(output_path, critter)
    lattice_file = os.path.join(output_path, critter_lattice)

    #convert long matrices to string
    #identify columns that are lists/matrics, etc.
    #test = summary.copy()

    object_columns = list(filter(lambda c: summary[c].dtype == 'O', summary.columns))
    for c in object_columns[1:]:
        summary[c] = list(map(lambda entry: entry.tolist()[0], summary[c]))

    #new_object_columns = list(filter(lambda c: summary[c].dtype == 'O', summary.columns))


    #include header only the first time we write to a file
    if not os.path.exists(critter_file):
        print(critter_file, 'does not exist')
        summary.to_csv(critter_file, mode='a', header=True)
    else:
        print(critter_file, 'exists')
        summary.to_csv(critter_file, mode= 'a', header= False)

    #write lattice data frame to file
    #note, this writes over any previous runs.
    #Be certain to store any data you want earlier
    lattice_df.to_csv(lattice_file, mode = 'w', header = True)



for i in range(1):
    print('run', i)
    main()
    i = 1+1

