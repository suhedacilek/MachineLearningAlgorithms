# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:45:50 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% create dataset

x1 = np.random.normal(25,5,1000) #normal Gaussian demek. 25 ortlamaya sahip 5 sigmaya sahip 1000 tane değer üret.
y1 = np.random.normal(25,5,1000)

x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x3), axis = 0)
y = np.concatenate((y1,y2,y3), axis = 0)

dictionary = {"x": x, "y": y}

data = pd.DataFrame(dictionary)

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()

## %% k means algoritması bunu görecek
#plt.scatter(x1,y1, color ="black")
#plt.scatter(x2,y2,color ="black")
#plt.scatter(x3,y3,color ="black")
#plt.show()

# %% K MEANS

from sklearn.cluster import KMeans

wcss = []

for k in range (1,15):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_) #wcss metricinin içine kmeans.inertia_ sını içine depola. İnertia = her bir k degeri icin wcss degerini bul demek.
    
plt.plot(range(1,15),wcss) #x degerlerim 1 den 15 e kadar, y degerlerim wcss degerlerim.
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()
plt.savefig("elbow.png")

# %% k = 3 modelim

kmeans2 = KMeans(n_clusters=3)
clusters = kmeans2.fit_predict(data)
data["label"] = clusters

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color = "red")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color = "green")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color = "blue")
plt.scatter(kmeans2.cluster_centers_[:,0], kmeans2.cluster_centers_[:,1], color = "yellow") # 2 boyutlu birseydir. Centroidimiz 0 ıncı x ekseni 1, y ekseni.
plt.show()
plt.savefig("centroid.png")























