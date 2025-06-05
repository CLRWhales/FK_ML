#this script seeks to apply the UMAP technique to explore and evaluate the latent space
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap



path = ''
data = pd.read_csv(path)