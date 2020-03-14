import networkx as nx
import numpy as np
import pandas as pd
from itertools import chain
from numpy.random import choice, rand
from scipy import sparse
from statistics import mean, stdev

#*******************************
#written to take in sparse adjacency matrices
#note, currently cannot accept negative weights
#*******************************

#Takes a sparse matrix of any graph and returns the sparse adjacency for the associated simple graph
def Adj_to_simple(adj):
    adjT= adj.transpose()
    A_sym = adj + adjT
    A_sym[A_sym>0]=1
    print('A_sym size', A_sym.shape, 'of type', type(A_sym))
    return A_sym

#takes in a sparse adjacency matrix and returns a degree distribution (matrix)
def degree_dist(adj):
    degree = adj.sum(0)#sorts along row, but shouldn't matter
    degree = np.sort(degree)
    return degree



#computes the eigenvalues of a (sparse) real symmetric square matrix (vertices \times vertices)
#returns matrix
#note, we are tossing the eigenvectors
def eigenvalue_list(adj, exclude=1):
    adj = adj.astype(float)
    verts = adj.shape[0]
    evals, evecs = sparse.linalg.eigsh(adj, k=verts-exclude, which='SA')
    evals = np.sort(evals)
    return np.asmatrix(evals)

#count number of triangles
#note: takes in sparse object
#output: needs to return a matrix, not a number
def triangle_count(adj):
    cubed = adj *adj * adj
    num_triangles = cubed.diagonal().sum()/ 6
    if not num_triangles.is_integer():
        print(num_triangles, 'is not integral')
    else:
        return np.asmatrix(num_triangles)


#Convolve models, turning models off if parameters become 0
def Convolve(vertices, parameters):
    MINFLOAT = 1e-15 #Set a minimum value to remove floating point errors
    prob = parameters[0]
    radius = parameters[1]
    connection = parameters[2]
    geom_prob = parameters[3]
    #All adjacencies are 0 (make certain they are sparse)
    A_SG = sparse.csr_matrix((vertices, vertices))  #soft geometric graph.
    A_ER = sparse.csr_matrix((vertices, vertices))
    A_G = sparse.csr_matrix((vertices, vertices))
    A_BA = sparse.csr_matrix((vertices, vertices))
    A_DP = sparse.csr_matrix((vertices, vertices)) #random dot product graph
    #turn on models one at a time
    if prob >= MINFLOAT:
        #Create Erdos Renyi
        ER = nx.fast_gnp_random_graph(vertices, prob)
        A_ER = nx.adjacency_matrix(ER)
    if radius >= MINFLOAT:
        #Create Geometric
        #G = nx.random_geometric_graph(vertices, radius)
        #A_G = nx.adjacency_matrix(G)
        #create soft geometric graph
        dist = lambda x: geom_prob
        SG = nx.soft_random_geometric_graph(vertices, radius, p_dist=dist)
        A_SG = nx.adjacency_matrix(SG)
    ####
    #code for a multi step soft geometric
    ####
    #if radius_large >= radius:
    #    #define a uniform probability for soft geometric
    #    def uniform(dist):
    #        return soft_prob
    if connection >= MINFLOAT:
        #Create Barabasi-Albert if m > 0; else turn BA off
        BA = nx.barabasi_albert_graph(vertices, connection)
        A_BA = nx.adjacency_matrix(BA)
    #if dimension>= MINFLOAT:
    #    #Create Random Dot product matrix
    #    rows = rand(dimension, vertices)
    #    A_DP = np.matmul(np.transpose(rows), rows)
    #    for i in range(vertices):
    #        A_DP[i,i] = 0
    #    A_DP[A_DP >= .5] = 1
    #    A_DP[A_DP < .5] = 0
    #    A_DP = sparse.csr_matrix(A_DP)

    #Convolve: add adjacency matrices; anything positive gets set to 1
    A = A_ER + A_G + A_BA + A_SG + A_DP
    #Check that adjacency matricx is symmetric
    #Should never see this error
    if not np.array_equal(A.toarray(), A.toarray().transpose()):
        print('Adjacency not symmetric, WARNING: NOT a simple graph')
    A[A>0]=1
    return A

#find the euclidean distance between two 1 \times n matrices or arrays
#both inputs should be arrays or matrices
def euclidean_distance(vec1, vec2):
    difference = vec1 - vec2
    distance= np.linalg.norm(difference)
    return distance

#generate appropriate sized sample of adjacency matrices given a set of parameters
#returns a list of sparse matrices
def generate_sample(vertices, parameters, sample_size):
    Adj_list = []
    while len(Adj_list)<sample_size:
        A = Convolve(vertices, parameters)
        Adj_list.append(A)
    return(Adj_list)

#matrix of characteristics (denoted by function) evaluated on a sample of graphs (each row represents the characteristic vector)
def characteristic_from_sample(sample, characteristic):
    #initialize matrix
    characteristic_matrix = characteristic(sample[0])
    sample_size = len(sample)
    #add on the last sample_size-1 rows
    for i in range(1, sample_size):
        characteristic_matrix = np.append(characteristic_matrix,characteristic(sample[i]), axis=0)
    return characteristic_matrix

#list of difference of orginal graph from sample along characteristic
#def sample_difference(sample, original):
#    difference_list = list(map(lambda v: original - v, sample))
#    return difference_list

#Calculate the mean and stdev degree distribution
#if sample_size is 1, stdev is defined as 0 for each vertex
#Recall: input is an np.matrix
#performs operations along column
#output: matrix sized as a row of characteristic sample
def moments_of_characteristic(characteristic_sample):
    #Recall: sample_size = characteristic_sample.shape[0]
    if characteristic_sample.shape[0] == 1:
        raise NameError("either sample size is 1 (so mean and stdev is meaningless) or this is a row vector")
    #gives a mean degree for each vertex based on random draws defined by convolution_model
    sample_mean = characteristic_sample.mean(0)
    sample_stdev = characteristic_sample.std(0)
    #if 1 by 1 outputs, retun number
    if sample_mean.shape == (1,1):
        sample_mean = sample_mean[0,0]
    if sample_stdev.shape == (1,1):
        sample_stdev = sample_stdev[0,0]
    return sample_mean, sample_stdev

#Calculate the mean and stdev degree distribution
#if sample_size is 1, stdev is defined as 0
#def triangle_moments(vertices, parameters, sample_size):
#    triangle_list = []
#    for i in range(sample_size):
#        A = Convolve(vertices, parameters)
#        triangle_list.append(triangle_count(A))
#    mean_triangle_count = mean(triangle_list)
#    if sample_size == 1:
#        stdev_triangle_count = 0
#    else:
#        stdev_triangle_count = stdev(triangle_list)
#    return mean_triangle_count, stdev_triangle_count


###############################
#Stuff for microns
###############################
#calculate overall density
#returns a matrix
def density(adj):
    vert_num = adj.shape[0]
    dd = degree_dist(adj)/(vert_num*(vert_num-1))
    return np.asmatrix(dd)

#calculate richclub coefficients
#returns a matrix 1 \times max_degree
def rich_club(adj):
    dd = degree_dist(adj)
    max_degree = dd[0, -1] #max degree is last element... can do this because sorted
    rich_club_coefs = np.asmatrix(np.zeros(max_degree))
    for k in range(max_degree):
        greater_degree_index = np.where(dd>k)[1][0] #again, can choose first element because sorted
        greater_degrees = dd[0, greater_degree_index:] #only bigger degrees
        n_k = len(greater_degrees)
        twice_e_k= greater_degrees.sum(0)
        rich_club_coefs[0,k] = twice_e_k/(n_k * (n_k-1))
    return rich_club_coefs

#local clustering coefficient
#note: takes in sparse object
#output: needs to return a matrix
def local_clustering(adj):
    local_coefficients = np.asmatrix(np.zeros(adj.shape[0]))
    for i in range(adj.shape[0]):
        #identify non-zero rows in ith column
        row = adj[i].todense()
        neighbor_list = list(np.where(row >0)[1])
        num_neigbors = len(neighbor_list)-1 #subtract 1 so as not to count self
        #keep only then non-zero columns
        skinny = adj.tocsr()[neighbor_list,:]
        #keep only the non-zero rows
        #this gives the adjacency for the closed neighborhood
        nbd_bar_adj = skinny.tocsc()[neighbor_list, :]
        #count local traingles
        num_triangles = triangle_count(nbd_bar_adj)[0,0]
        coefficient = 2 * num_triangles/ (num_neigbors * (num_neigbors-1))
        local_coefficients[0,i] = coefficient
    return np.sort(local_coefficients)





