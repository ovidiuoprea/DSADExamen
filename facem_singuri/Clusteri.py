import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from scikitplot.metrics import plot_silhouette
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples


def nan_replace_t(t: pd.DataFrame):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if is_numeric_dtype(t[v]):
            t.fillna({v: t[v].mean()}, inplace=True)
        else:
            t.fillan({v: t[v].mode()[0]}, inplace=True)

def metoda_elbow(ierarhie, nr_clusteri=None):
    n = ierarhie.shape[0] + 1
    if nr_clusteri is not None:
        distante = ierarhie[1, 2] - ierarhie[n-2, 2]
        nr_jonctiuni = np.argmax(distante) + 1
        nr_clusteri = n - nr_jonctiuni
    else:
        nr_jonctiuni = n - nr_clusteri
    threshold = (ierarhie[nr_jonctiuni, 2] + ierarhie[nr_jonctiuni - 1, 2]) / 2
    return nr_clusteri, threshold

def 


set_date = pd.read_csv("../data_in/Cluster-Mortalitate.csv", index_col=0)
nan_replace_t(set_date)
variabile = list(set_date)

x = set_date[variabile].values

ierarhie = linkage(y, "ward")