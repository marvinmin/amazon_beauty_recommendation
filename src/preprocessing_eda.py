import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from IPython.core.display import display, HTML
from scipy.sparse import csr_matrix as sparse_matrix

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

def create_X(ratings, n, d, user_key="user", item_key="item"):
    """
    Creates a sparse matrix using scipy.csr_matrix and mappers to relate indexes to items' id.
    
    Parameters:
    -----------
    ratings: pd.DataFrame
        the ratings to be stored in the matrix;
    n: int
        the number of items
    d: int
        the number of users
    user_key: string
        the column in ratings that contains the users id
    item_key: string
        the column in ratings that contains the items id
    
    Returns: (X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind)
    --------
    X: np.sparse
        the sparse matrix containing the ratings.
    user_mapper: dict
        stores the indexes of the users - the user_id is the key;
    item_mapper: dict
        stores the indexes of the items - the item_id is the key;
    user_inverse_mapper: dict
        stores the user id - the user index is the key;
    item_inverse_mapper: dict
        stores the item id - the item index is the key;
    user_ind: list
        indexes of the users (in the order they are in ratings);
    item_ind: list
        indexes of the items;
    """
    
    user_mapper = dict(zip(np.unique(ratings[user_key]), list(range(d))))
    item_mapper = dict(zip(np.unique(ratings[item_key]), list(range(n))))

    user_inverse_mapper = dict(zip(list(range(d)), np.unique(ratings[user_key])))
    item_inverse_mapper = dict(zip(list(range(n)), np.unique(ratings[item_key])))

    user_ind = [user_mapper[i] for i in ratings[user_key]]
    item_ind = [item_mapper[i] for i in ratings[item_key]]

    X = sparse_matrix((ratings["Rating"], (item_ind, user_ind)), shape=(n,d))
    
    return X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind

def get_id(X, inverse_mapper):
    """
     Gets the items IDs which have the most reviews, the most total ratings and the lowest 
     average ratings from the sparse matrix 
    
    Parameters:
    -----------
    X: np.sparse
        the sparse matrix containing the ratings
    inverse_mapper: dict
        stores the item id - the item index is the key
    
    Returns: (item_id_most_reviews, item_id_most_total, item_id_lowest_avg)
    --------
    item_id_most_reviews: str
        the ID of item with the most reviews
    item_id_most_total: str
        the ID of item with the most total ratings
    item_id_lowest_avg: str
        the ID of item with the lowset average ratings
    """
    # Get the item with the most reviews
    ind_most_reviews = np.argmax(X.getnnz(axis=1))
    item_id_most_reviews = inverse_mapper[ind_most_reviews]

    # Get the item with the most total stars
    ind_most_total= np.argmax(np.sum(X, axis = 1))
    item_id_most_total = inverse_mapper[ind_most_total]

    # Get the item with the lowest average stars
    ind_lowest_avg= np.argmin(np.squeeze(np.sum(X, axis = 1)) / X.getnnz(axis=1))
    item_id_lowest_avg = inverse_mapper[ind_lowest_avg]
    
    return item_id_most_reviews, item_id_most_total, item_id_lowest_avg

def plot_hist(X, key="item"):
    """
    Make two histgrams of the number of ratings per user and the number of ratings per item.
    
    Parameters:
    -----------
    X: np.sparse
        the sparse matrix containing the ratings
    key: str (optional)
        the name of the item we are interested in
    """   
    plt.figure(0, figsize=(6, 6))
    plt.hist(X.getnnz(axis=0),bins = 100)
    plt.yscale('log', nonposy='clip')
    plt.title("Histogram of the number of ratings per user", size = 20)
    plt.xlabel("The number of ratings", size = 16)
    plt.ylabel("Count (log scaled)", size = 16)
    plt.show();
    plt.figure(0, figsize=(6, 6))
    plt.hist(X.getnnz(axis=1),bins = 100)
    plt.yscale('log', nonposy='clip')
    plt.title("Histogram of the number of ratings per " + key, size = 20)
    plt.xlabel("The number of ratings", size = 16)
    plt.ylabel("Count (log scaled)", size = 16)
    plt.show();

def fit_nn(X, vec, num_neighbor, metric = 'euclidean'):
    """
    Fits a nearest neighbors model and finds the nearest neighbors for a given item.
    
    Parameters:
    -----------
    X: np.sparse
        the sparse matrix containing the ratings.
    vec: np.sparse
        the sparse matric with one row containing the ratings of the item 
        whose neighbors to be found.
    num_neighbor: int
        the number of neighbors
    metric: string (optional)
        the distance metric for finding the neighbors
    
    Returns: neighbors_ind
    --------
    neighbors_ind: list
        indexes of the neighbors (include the original item itself)
    """
    nn_model = NearestNeighbors(metric=metric)
    nn_model.fit(X)
    neighbors_ind = nn_model.kneighbors(vec, num_neighbor, return_distance=False)
    return neighbors_ind

def print_result(neighbors_ind, item_inverse_mapper):
    """
    Prints the neighbors found by fit_nn()
    
    Parameters:
    -----------
    neighbors_ind: list
        indexes of the neighbors (include the original item itself) found by fit_nn()
    item_inverse_mapper: dict
        stores the item id - the item index is the key
    """
    print("The 6 items most similar to the most reviewed product (exclusive) are:")
    for item_ind in np.squeeze(neighbors_ind)[1:]:
        item_id = item_inverse_mapper[item_ind]
        display(HTML('<a href="%s">%s</a>' % ('https://www.amazon.com/dp/'+item_id, 
                                              item_id)))

def cal_total_popularity(X, index, item_inverse_mapper):
    """
    Calculates and prints the total popularity of the neighbors found by fit_nn()
    
    Parameters:
    -----------
    X: np.sparse
        the sparse matrix containing the ratings.
    index: list
        indexes of the neighbors (include the original item itself) found by fit_nn()
    item_inverse_mapper: dict
        stores the item id - the item index is the key
    """
    print("The total popularity:")
    for item_ind in np.squeeze(index):
        item_id = item_inverse_mapper[item_ind]
        display(HTML('<a href="%s">%s</a>' % ('https://www.amazon.com/dp/'+item_id, 
                                              item_id)))
        print("Total stars: {}".format(int(X[item_ind,:].sum()))) 

