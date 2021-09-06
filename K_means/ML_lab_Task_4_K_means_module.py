#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[22]:


class k_means():
    def __init__(self, num_centroids, iterations):
        '''
        num_centroids - number of clusters expected
        iterations - number of iterations to form clusters
        '''
        self.num_centroids = num_centroids
        self.iterations = iterations
        self.clusters = []
           
    def _get_centroids(self, iteration):
        if iteration == 0:
            mean = [5, 9]
            cov = [[2.5, 0.8], [0.8, 0.5]]
            self.centroids = np.random.multivariate_normal(mean, cov, self.num_centroids).T
            self.centroids = self.centroids.T
        else:
            columns = ['Iteration','Obs_num','Coor_1','Coor_2','Cluster_num','Cluster_dist']
            df_new_centroids = pd.DataFrame(self.clusters, columns = columns).groupby(by = ['Cluster_num','Iteration'])['Coor_1','Coor_2'].mean()
            #print(df_new_centroids)
            df_new_centroids.reset_index(inplace=True)
            self.centroids = np.array(df_new_centroids[df_new_centroids['Iteration'] == (iteration-1)][['Coor_1','Coor_2']])
           
        return self.centroids
    
    def train(self,data):
        self.data = data
        for iteration in range(self.iterations):
            #print(iteration)
            centroids = self._get_centroids(iteration)
            distances = self._distance_calc(centroids)
            self.clusters = self._find_cluster(iteration, distances)
        
        return self.clusters
            #print(clusters)    
        
    def _distance_calc(self, centroids):
        d = {}
        for i in range(len(self.data)):
        
            d[i] = {}
            for centroid in range(len(centroids)):
                d[i][centroid] = {}
                d[i][centroid] = sum((self.data[i] - centroids[centroid])**2)**0.5
        return d

    def _find_cluster(self, iteration, dist):
    
        '''
        iteration - number of iteration 
        x - initial data 
        dist - dictionary with calculated distances between each point and suggessted clusters 
        '''

        # iterating over each data point (observation) and for each observation taking
        # the closest cluster for the iteration coordinates of the observation 
        
        for i in range(len(self.data)):
            # the closest cluster to the observation
            i_cluster = min(dist[i], key=dist[i].get)
            # distance of the observationb to the closest cluster 
            i_dist = min(dist[i].values())
            self.clusters.append([iteration, i, self.data[i][0], self.data[i][1], i_cluster, i_dist])
        
        return self.clusters

    #def loss(self, clusters):
    # TBD
            


# In[ ]:




