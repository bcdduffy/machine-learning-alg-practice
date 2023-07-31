# Standard imports
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn
class CustomKMeans():
    # Initialize all attributes
    def __init__(self, k):
        self.k_ = k # Number of clusters
        self.labels_ = 0 # Each sample's cluster label
        self.inertia_ = 0 # Sum of all samples' distances from their centroids
    # Find K cluster centers & label all samples
    def fit(self, data, plot_steps=False):
        # Fit the PCA module & Transform our data for later graphing
        #print("HHHHHHHHh")
        self.pca = PCA(2).fit(data)
        self.data = pd.DataFrame(data)
        self.data_pca = pd.DataFrame(self.pca.transform(data))
        self.data_pca.columns = ['PC1', 'PC2']
        # Initialize variables
        self.iteration = 1
        n = data.shape[0]
        # Initialize centroids to random datapoints
        self.centroids = data.iloc[np.random.choice(range(n), self.k_, replace=False)].copy()
        self.centroids.index = np.arange(self.k_)


        last_inertia = 1
        inertia_diff = 1
        while(inertia_diff != 0 ):
            self.labels_, self.inertia_ = self.update_clusters_intertia(data)
            self.centroids = self.centroid_updates(data)
            #print(last_inertia - self.inertia_)
            inertia_diff = last_inertia - self.inertia_
            last_inertia = self.inertia_
            if plot_steps:
                self.plot_state()
                self.iteration += 1
        #self.inertia_ = self.inertia_ - last_inertia
        # while (not converged): # psuedocode - up to you to implement stopping criterion
        #print(data.shape())
        #self.labels_ = [random.randint(0,self.k_-1) for i in range(n)] # Update
    #self.inertia_ = np.sum(np.arange(n)) # Update
    # show data & centroids at each iteration when testing performance


        return self

# Plot projection of data and centroids in 2D
    def plot_state(self):
        # Project the centroids along the principal components
        centroid_pca = self.pca.transform(self.centroids)
        # Draw the plot
        plt.figure(figsize=(8,8))
        plt.scatter(self.data_pca['PC1'], self.data_pca['PC2'], c=self.labels_)
        plt.scatter(centroid_pca[0], centroid_pca[1], marker = '*', s=1000)
        plt.title("Clusters and Centroids After step {}".format(self.iteration))
        plt.show()
    def update_clusters_intertia(self, data):
        all_dist_centroids = sklearn.metrics.pairwise_distances(data, self.centroids)
        all_distances = np.amin(all_dist_centroids,axis=1)
        #print(all_distances)
        square_dist = np.square(all_distances)
        self.labels_ = np.argmin(all_dist_centroids,axis=1)
        self.inertia_ = np.sum(square_dist)
        #print(sum_square)
        return self.labels_, self.inertia_
    def centroid_updates(self, data):
        all_clust_range = range(self.k_)
        avg_dist = np.array([np.mean(data[self.labels_==entry], axis=0) for entry in all_clust_range])
        self.centroids = avg_dist
        return self.centroids
