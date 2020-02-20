#returns a sparse adjancy matrix for a variety of data types
import networkx as nx
from scipy import sparse
import numpy as np

#take .graphml and returns a sparse array adjacency matrix
def graph_to_Adj(graph):
    #read in file as dictionary
    g = nx.read_graphml(graph)
    #remove self loops
    selfloops = list(g.selfloop_edges())
    g.remove_edges_from(selfloops)
    #extract adjacency matrix (ignoring node names)
    A_gw = nx.adjacency_matrix(g)
    return A_gw

#take a .csv and returns a sparse matrix
def csv_to_Adj(csv):
    #read in file as array
    A_gw = np.genfromtxt(csv, delimiter= ',')
    #clean data step that we should get rid of once we enforce clean data
    A_gw = A_gw[1: , 1:]
    #remove self loops (change diagonals to 0)
    np.fill_diagonal(A_gw, 0)
    A_gw = sparse.csr_matrix(A_gw)
    return A_gw

def convert(data):
    #data = input('the .csv or .graphml file path is ')
    #if the data is a graphml file
    if data[-8:] == '.graphml':
        Adj = graph_to_Adj(data)
    elif data[-4:] == '.csv':
        Adj = csv_to_Adj(data)
    else:
        print('wrong file type')
    print('file converted, Adj has size', Adj.shape)
    return Adj
        

