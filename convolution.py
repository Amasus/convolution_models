import networkx as nx
import numpy as np
import pandas as pd

from itertools import product
from numpy.random import choice, random
from scipy import sparse
import graph_functions as gf

#takes parameters and returns a dictionary of the stats that will be recorded in dataframe. Returns a dictionary or numbers, 1 tuple of three, and a bunch of 1 by n matrices
def extract_sample_stats(vertex_num, params, sample_size, orig_dd, orig_ev, tri_num):
    #generate sample
    sample = gf.generate_sample(vertex_num, params, sample_size)

    #extract deg dist, triangle number, eigenvalues from sample

    #calculate stuff for degree distribution (dd)
    sample_dds = gf.characteristic_from_sample(sample, gf.degree_dist) #list of degree distributions (matrices)
    mean_dd_vec, stdev_dd_vec = gf.moments_of_characteristic(sample_dds) #means and stdevs of each entry
    sample_distances_from_mean_dd_vec = np.asmatrix([gf.euclidean_distance(degree_dist, mean_dd_vec) for degree_dist in sample_dds]) #list of distances of each sample from mean degree distribution
    mean_distance_dd, stdev_distance_dd = gf.moments_of_characteristic(np.transpose(sample_distances_from_mean_dd_vec)) # mean and stdev of distances of deg_dist vector from mean
    orig_dd_distance_from_mean_dd =  gf.euclidean_distance(orig_dd, mean_dd_vec) #distance of original graph from sample mean
    sample_distances_from_original_dd = np.asmatrix([gf.euclidean_distance(degree_dist, orig_dd) for degree_dist in sample_dds]) #list of distances of each sample from original degree distribution
    mean_distance_from_orig_dd, stdev_distance_from_orig_dd = gf.moments_of_characteristic(
        np.transpose(sample_distances_from_original_dd))  # mean and stdev of distances of deg_dist vector from original

    #calculate stuff for the triangle count
    sample_triangle_counts =  np.transpose(gf.characteristic_from_sample(sample, gf.triangle_count)) #list of triangle counts
    mean_triangle_num, stdev_triangle_num = gf.moments_of_characteristic(np.transpose(sample_triangle_counts)) #means and stdev

    #calculate stuff for the eigenvalues (ev)
    sample_evs = gf.characteristic_from_sample(sample, gf.eigenvalue_list)  # list of eivenvlues (matrices)
    mean_ev_vec, stdev_ev_vec = gf.moments_of_characteristic(sample_evs)  # means and stdevs of each entry
    sample_distances_from_mean_ev_vec = np.asmatrix([gf.euclidean_distance(eigenvalues,  mean_ev_vec) for eigenvalues in
                                  sample_evs])  # list of distances of each sample from mean eigenvalues
    mean_distance_ev, stdev_distance_ev = gf.moments_of_characteristic(np.transpose(sample_distances_from_mean_ev_vec))  # mean and stdev of distances of eigenvalue vector from origin
    orig_ev_distance_from_mean_ev = gf.euclidean_distance(orig_ev, mean_ev_vec)
    sample_distances_from_original_ev = np.asmatrix([gf.euclidean_distance(degree_dist, orig_ev) for degree_dist in sample_evs]) #list of distances of each sample from original evals
    mean_distance_from_orig_ev, stdev_distance_from_orig_ev = gf.moments_of_characteristic(
        np.transpose(sample_distances_from_original_ev))  # mean and stdev of distances of deg_dist vector from original


    #calculate error function
    #vector to minimize (orig_distance_from_mean_dd, orig_distance_from_mean_ev)
    #note error 1 is floats, while error 2 is matrices
    error1 = (orig_ev_distance_from_mean_ev**2 + orig_dd_distance_from_mean_dd**2)**.5
    error2 = (mean_distance_from_orig_dd*2 + mean_distance_from_orig_ev**2)**.5 #if usuing this error, rank original distance from mean as orig dist from orig = 0
    reporting_dict = {'Coords': params, 'original dd vec': orig_dd, 'vector mean of deg dist': mean_dd_vec, 'vector stdev of deg dist': stdev_dd_vec,
                      'sample distances from mean dd vec': sample_distances_from_mean_dd_vec, 'mean distance from mean_dd' : mean_distance_dd,
                      'stdev distance from mean_dd' : stdev_distance_dd, 'original dd distance from mean': orig_dd_distance_from_mean_dd,
                      'distance of sample from original dd': sample_distances_from_original_dd, 'mean distance from original dd': mean_distance_from_orig_dd,
                      'stdev distance from original dd': stdev_distance_from_orig_dd, 'original triangles': tri_num[0,0], 'sample triangle counts': sample_triangle_counts,
                      'mean triangle count': mean_triangle_num, 'stdev triangle count': stdev_triangle_num, 'original ev vec': orig_ev,
                      'vector mean of eigenvalues': mean_ev_vec, 'vector stdev of eigenvalues': stdev_ev_vec,'sample distances from mean ev vec': sample_distances_from_mean_ev_vec,
                      'mean distance from mean_ev' : mean_distance_ev, 'stdev distance from mean_ev' : stdev_distance_ev,
                      'original ev distance from mean': orig_ev_distance_from_mean_ev, 'distance of sample from original ev': sample_distances_from_original_ev,
                      'mean distance from original ev': mean_distance_from_orig_ev,
                      'stdev distance from original ev': stdev_distance_from_orig_ev, 'error': error1}
    #bad = list(filter(lambda k: (not (isinstance(reporting_dict[k], int) or isinstance(reporting_dict[k], float) or isinstance(reporting_dict[k], tuple) or
                                      #isinstance(reporting_dict[k], np.matrix))) or (isinstance(reporting_dict[k], np.matrix) and reporting_dict[k].shape[1]== 1), reporting_dict))

    return reporting_dict



#####TODO:COPYING CODE AND KILLING PUPPIES!!!!! REFACTOR
def convolution_metrics_step(Adj, step_size_multiplier, sample_size= 1):

    #extract facts about the connectome, henceforth called the original graph
    #deg_dist, triangle number, eigenvalues
    vertex_num = Adj.shape[0]
    #convert to a simple graph matrix (symmetric, entries 0, 1)
    Adj = gf.Adj_to_simple(Adj)
    orig_deg_dist = gf.degree_dist(Adj)
    num_half_edges = orig_deg_dist.sum(1)[0,0]
    edge_density = num_half_edges/ (vertex_num * (vertex_num -1))
    radius = (edge_density/(4* np.pi))**.5
    orig_tri_num = gf.triangle_count(Adj)
    orig_eigenvalues = gf.eigenvalue_list(Adj)
    print('connectome has ', orig_tri_num[0,0], 'triangles')

    #set seed
    candidate_coord = (edge_density/2,radius/2, 2)

    #extract stats for coordinate
    sample_stats = extract_sample_stats(vertex_num, candidate_coord, sample_size, orig_deg_dist, orig_eigenvalues, orig_tri_num)

    #now calculated in sample_stats
    old_error = sample_stats['error']

    #start simultaneous minimization
    new_error = 0
    p_step = edge_density * step_size_multiplier
    r_step = radius * step_size_multiplier
    m_step = 1


    while new_error < old_error:
        if new_error== 0: #initial round or perfect fit
            old_error = old_error #skip this step
        else: #update error
            old_error = new_error

        #initialize dataframe
        df = pd.DataFrame()
        df = df.append(sample_stats, ignore_index=True)

        #read off coordinates from last best
        (p, r, m) = df['Coords'].item()
        # initialize dataframe


        #take a step in each direction. List coordinates
        coords =  [(min({p+p_step, edge_density}), r, m), (max({0, p - p_step}), r, m) ,(p, min({r+r_step, radius}), m), (p, max({0, r - r_step}), m), (p, r,m+m_step), (p,r, max({0, m - m_step}))]
        for prm in coords:

            #comutue stats for each coordinate. append to data frame
            prm_stats = extract_sample_stats(vertex_num, prm, sample_size, orig_deg_dist, orig_eigenvalues, orig_tri_num)
            df = df.append(prm_stats, ignore_index= True)

        # select smallest error
        new_error = df['error'].min()  # smallest error
        #extract stats with least error
        sample_stats = df.loc[df['error'] == new_error]
        print("error =", sample_stats['error'].item(), "parameters = ", sample_stats['Coords'].item())
    summary = pd.DataFrame()
    summary = summary.append(sample_stats, ignore_index= True)
    summary['sample_size'] = sample_size
    summary['edge density'] = edge_density
    summary['radius'] = radius
    summary['number of vertices'] = vertex_num
    return summary

def convolution_metrics_lattice(Adj, step_size_multiplier, sample_size= 1):

    #extract facts about the connectome, henceforth called the original graph
    #deg_dist, triangle number, eigenvalues
    vertex_num = Adj.shape[0]
    #convert to a simple graph matrix (symmetric, entries 0, 1)
    Adj = gf.Adj_to_simple(Adj)
    orig_deg_dist = gf.degree_dist(Adj)
    num_half_edges = orig_deg_dist.sum(1)[0,0]
    edge_density = num_half_edges/ (vertex_num * (vertex_num -1))
    radius = (edge_density/(4* np.pi))**.5
    orig_tri_num = gf.triangle_count(Adj)
    orig_eigenvalues = gf.eigenvalue_list(Adj)
    print('connectome has ', orig_tri_num[0,0], 'triangles')

    # initialize dataframe
    df = pd.DataFrame()

    #set the lattice.
    #note that this is currently an iterable, and will be consumed after the loop
    #recall that np.arange is an half empty interval [start, stop)
    p_points = np.arange(0, edge_density, step_size_multiplier*edge_density)
    r_points = np.arange(0, radius, step_size_multiplier* radius)
    m_points = range(10)
    lattice_points = product(p_points, r_points, m_points)

    for (p,r,m) in lattice_points:
        #extract stats from current coordinate
        current_coord = (p,r,m)
        sample_stats = extract_sample_stats(vertex_num, current_coord, sample_size, orig_deg_dist, orig_eigenvalues,
                                    orig_tri_num)
        print("error =", sample_stats['error'], "parameters = ", sample_stats['Coords'])
        #append to dataframe
        df = df.append(sample_stats, ignore_index=True)


    # select smallest error
    min_error = df['error'].min()  # smallest error
    candidate_stats = df.loc[df['error'] == min_error]



    summary = pd.DataFrame()
    summary = summary.append(candidate_stats, ignore_index= True)
    summary['sample_size'] = sample_size
    summary['edge density'] = edge_density
    summary['radius'] = radius
    summary['number of vertices'] = vertex_num
    return summary, df
        

        
