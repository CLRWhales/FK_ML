#this is a look at UMAP for the speed selected files
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
from sklearn.metrics import mean_squared_error
import os
# UMAP
import umap.umap_ as umap

#%%#this looks at the reg vs true values
names = pd.read_csv("C:\\Users\\Calder\\Models\\FK_ver3_subset_transform_20250622T182320\\names.csv")
regest = pd.read_csv("C:\\Users\\Calder\\Models\\FK_ver3_subset_transform_20250622T182320\\regest.csv")
regtrue = pd.read_csv("C:\\Users\\Calder\\Models\\FK_ver3_subset_transform_20250622T182320\\regtrue.csv")
latents = pd.read_csv("C:\\Users\\Calder\\Models\\FK_ver3_subset_transform_20250622T182320\\latent_states.csv")

#%%
whalelist = [['FW',"D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640\\FK\\20220821T183537Z\\T0_X512_F44_K316_V2304.0_20220821T183537Z.png"],
             ['FW',"D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640\\FK\\20220821T183537Z\\T0_X768_F43_K192_V2112.0_20220821T183537Z.png"],
             ['DS',"D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640\\FK\\20220821T183537Z\\T0_X1536_F94_K106_V1945.6_20220821T183537Z.png"],
             ['FW',"D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640\\FK\\20220821T183537Z\\T256_X512_F38_K325_V1736.3478260869565_20220821T183537Z.png"],
             ['FW',"D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640\\FK\\20220821T183537Z\\T256_X768_F39_K179_V1595.844155844156_20220821T183537Z.png"],
             ['FW',"D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640\\FK\\20220821T183537Z\\T512_X512_F39_K332_V1616.842105263158_20220821T183537Z.png"],
             ['FW',"D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640\\FK\\20220821T183537Z\\T512_X768_F41_K130_V1024.0_20220821T183537Z.png"],
             ['DS',"D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640\\FK\\20220821T183537Z\\T768_X1536_F86_K118_V1936.695652173913_20220821T183537Z.png"]]

idx = []
clas = []
for w in whalelist:
    name = os.path.basename(w[1])
    clas.append(w[0])
    idx.append(names[names.iloc[:,0]==name].index.tolist())
idx = np.concat(idx)
#%% This looks at columnwise amounts of MSE to see how each of the regressors perform

mse_dict = {}
for col in regest.columns.intersection(regtrue.columns):
    mse = mean_squared_error(regest[col], regtrue[col])
    mse_dict[col] = mse

# Result
print(mse_dict)
plt.figure()
plt.bar(range(len(mse_dict)), list(mse_dict.values()), align='center')
plt.xticks(range(len(mse_dict)), list(mse_dict.keys()))
plt.ylabel('MSE')
plt.show()

#%%

pca = PCA(n_components=2)
X_pca = pca.fit_transform(latents)

# Convert to data frame
principal_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
#%%
plt.figure(figsize=(8,6))
plt.scatter(principal_df.iloc[:,0], principal_df.iloc[:,1],alpha = 0.01, s = 0.1)
plt.title('PCA plot in 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(principal_df.iloc[idx,0], principal_df.iloc[idx,1],color = 'red', s = 2)

plt.figure(figsize=(8,6))
plt.scatter(principal_df.iloc[:,0], principal_df.iloc[:,1],c = regest.iloc[:,0], s = 0.1)
plt.title('PCA plot in 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()


plt.figure(figsize=(8,6))
plt.hexbin(principal_df.iloc[:,0], principal_df.iloc[:,1])
plt.title('PCA plot in 2D')
plt.colorbar()
plt.clim(10,200)
plt.xlabel('PC1')
plt.ylabel('PC2')

# %%
um = umap.UMAP()
X_fit = um.fit(latents)           # we'll use X_fit later
X_umap = um.transform(latents)

# Convert to data frame
umap_df = pd.DataFrame(data = X_umap, columns = ['umap comp. 1', 'umap comp. 2'])
fname = 'C:\\Users\\Calder\\Models\\FK_ver3_subset_transform_20250622T182320\\umap.csv'
umap_df.to_csv(fname)


#%%
umap_df = pd.read_csv("C:\\Users\\Calder\\Models\\FK_ver3_subset_transform_20250622T182320\\umap.csv")
plt.figure(figsize=(8,6))
plt.hexbin(umap_df.iloc[:,1], umap_df.iloc[:,2])
# plt.clim(0,200)
plt.colorbar()
plt.title('UMAP density in 2D')
plt.xlabel('umap component 1')
plt.ylabel('umap component 2')

plt.figure(figsize=(8,6))
plt.scatter(umap_df.iloc[:,1], umap_df.iloc[:,2],alpha = 0.01, s = 0.1)
plt.title('UMAP density in 2D')
plt.xlabel('umap component 1')
plt.ylabel('umap component 2')
plt.scatter(umap_df.iloc[idx,1], umap_df.iloc[idx,2], s = 5, color = 'red')


plt.figure(figsize=(8,6))
plt.scatter(umap_df.iloc[:,1], umap_df.iloc[:,2],c = regest.iloc[:,0], s = 0.1)
plt.title('UMAP density in 2D')
plt.xlabel('umap component 1')
plt.ylabel('umap component 2')
plt.colorbar()
# %%
