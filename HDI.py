
# coding: utf-8
from __future__ import print_function
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



data =  pd.read_csv("data.csv", delimiter = ',')

hdi=data[["HDI","Rank"]].values
range_n_clusters = [3, 4, 5, 6]

estimators={}
for n in range_n_clusters:
    estimators['k_means_'+str(n)]= KMeans(n_clusters=n)
     
              
X = hdi
fignum = 1

silhouette_table={}

for name, est in estimators.items():

    fig = plt.figure(fignum, figsize=(10, 7.5))
    plt.clf()#clear figure

    est.fit(X)

    labels = est.labels_
    cluster_labels = est.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_table[name]=silhouette_avg

    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        
    plt.scatter(X[:,0], X[:,1], c=labels.astype(np.float))
    centers = est.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1],marker='o', c="white", alpha=1, s=200)
    for i, c in enumerate(centers):
        plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)
        
    plt.annotate('average silhouette value: '+str(silhouette_table[name]),
                 xy=(0.4, 0.1), xytext=(0.4, 0.1))
            
    fignum = fignum + 1
    plt.title(name)
    plt.ylabel('Rank')
    plt.xlabel('HDI')
    plt.savefig('hdi'+str(fignum-1))


t=pd.DataFrame(data=silhouette_table.items(),columns=['Date', 'DateValue'])
print(t)