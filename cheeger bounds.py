import networkx as nx
import numpy as np
from scipy import sparse

import graph_functions as gf
from file_conversion import convert

#takes in a sparse adjacency and returns the laplacian
def adjacency_to_laplacian(adj):
    deg_dist_array = np.array(adj.sum(0))
    #return the matrix as a csr, as I think that is what everything else is
    diagonal_mat = sparse.diags(deg_dist_array, [0], format = 'csr')
    laplacian = diagonal_mat - adj
    return laplacian

#computes the interval of edge expansion bounds
def edge_explansion_bounds(evals):
    #find \lambda_2 (first non-zero eigenval)
    MIN_FLOAT = 10**(-8)
    i=0
    try:
        while evals[i]<MIN_FLOAT:
            i= i+1
        lambda2 = evals[i]
        lower_bound = lambda2/2
        upper_bound = np.sqrt(2*lambda2)
    except IndexError:
        print('all eigenvalues 0')
        lower_bound = None
        upper_bound = None
    return lower_bound, upper_bound

#This is the set of graphs/adjacencies we are interested in
#these files are all in ../data/
data_files_list = ['ZachKarateClub.csv', 'c.elegans.herm_pharynx_1.graphml', 'drosophila_medulla_1.graphml', 'drosophila_mushroom_body.csv']#,'mouse_retina_1.graphml']


#convert files to adjacency matrices (may be a MultiDigraph)
adjacency_list= [convert('../data/' + file) for file in data_files_list]

#convert adjacencies to simple graphs adjs
#this is a list of sparse matrices
simple_adjacency_list = [gf.Adj_to_simple(Adj) for Adj in adjacency_list]

#compute the list of Laplacians associated to these adjs
#this is a list of sparse matrices
laplacian_list = [adjacency_to_laplacian(Adj) for Adj in simple_adjacency_list]

#compute eigenvalues of the Adjacency matrices
#second argument indicates that we only want the first few values
#this is a list of 1 \time ver_num matrices
adj_eigen_val_list = [gf.eigenvalue_list(Adj, exclude=Adj.shape[0]-25) for Adj in adjacency_list]

#compute the eigenvalues of the Laplacian matrices
#this is a list of 1 \times vert_num matrices
lap_eigen_val_list = [gf.eigenvalue_list(Adj, exclude=Adj.shape[0]-25) for Adj in laplacian_list]

#type convert laplacian eigen values to a list of lists:
lap_eigen_val_list = [np.array(evals).flatten().tolist() for evals in lap_eigen_val_list]

#check that laplacian e-vals positive
MIN_FLOAT = 10**(-8)
positive_evals = [evals[0]>-1 * MIN_FLOAT for evals in lap_eigen_val_list]
if False in positive_evals:
    raise NameError('Negative Laplacian Eigenvalue!')

#find expansion bounds
#note the need for type conversions here
expansion_bounds = [edge_explansion_bounds(np.array(evals).flatten().tolist()) for evals in lap_eigen_val_list]

print('now what')

