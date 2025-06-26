#this script seeks to apply the UMAP technique to explore and evaluate the latent space
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# UMAP
import umap.umap_ as umap

#%%

path = "D:\\DAS\\FK\\meta_test_hr_20250620T162109\\meta.csv"
data = pd.read_csv(path)

data_sub = data.drop(columns=['Xoff', 'Toff'])
#SS=StandardScaler()
#X=pd.DataFrame(SS.fit_transform(data_sub), columns=data_sub.columns)
X = data_sub

plt.figure()
plt.hist(X['F']*128,bins = 128)
plt.figure()
plt.hist(X['K']*512, bins = 100)
plt.figure()
plt.hist(X['V'], bins = 100)

plt.figure()
plt.hexbin(X['F']*128,X['V']*10000)
plt.xlabel('frequency')
plt.ylabel('velocity of max energy')
plt.colorbar()
plt.clim(0,40)


plt.figure()
plt.hexbin(X['K'],X['F']*128)
plt.colorbar()
plt.clim(0,40)
plt.hlines(y = 5,xmin = 0, xmax = 1, color = 'red')

calls = X[(X['V']*10000 > 800)&(X['V']*10000 < 4000)]

plt.figure()
plt.hexbin(calls['K'],calls['F'])
plt.colorbar()
plt.clim(0,20)

plt.figure()
plt.hist(calls['F']*128,bins = 128)
plt.ylim(0,4000)
# %% trying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Convert to data frame
principal_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])

plt.figure(figsize=(8,6))
plt.scatter(principal_df.iloc[:,0], principal_df.iloc[:,1])
plt.title('PCA plot in 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.figure(figsize=(8,6))
plt.hexbin(principal_df.iloc[:,0], principal_df.iloc[:,1])
plt.title('PCA plot in 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')

#%% UMAP

um = umap.UMAP()
X_fit = um.fit(X)           # we'll use X_fit later
X_umap = um.transform(X)

# Convert to data frame
umap_df = pd.DataFrame(data = X_umap, columns = ['umap comp. 1', 'umap comp. 2'])



#%%
plt.figure(figsize=(8,6))
plt.hexbin(umap_df.iloc[:,0], umap_df.iloc[:,1])
plt.title('UMAP density in 2D')
plt.xlabel('umap component 1')
plt.ylabel('umap component 2')

plt.figure(figsize=(8,6))
plt.scatter(umap_df.iloc[:,0], umap_df.iloc[:,1])
plt.title('UMAP density in 2D')
plt.xlabel('umap component 1')
plt.ylabel('umap component 2')

# %%
# KMeans
kmeans = KMeans(n_clusters=10, n_init=15, max_iter=500, random_state=0)

# Train and make predictions
clusters = kmeans.fit_predict(X)

# Cluster centers
centroids = kmeans.cluster_centers_
#centroids_pca = pca.transform(centroids)

#%%
plt.figure()
plt.scatter(umap_df.iloc[:,0], umap_df.iloc[:,1], c = clusters)

#%%

plt.figure()
plt.scatter(data['timestamp'],data['xoff'],c = clusters)
plt.colorbar()
# %%
