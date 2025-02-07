import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from utils import nan_replace_t

set_date = pd.read_csv("data_in/Cluster-Mortalitate.csv", index_col=0)
nan_replace_t(set_date)
variabile = list(set_date)

# Implementare metoda Elbow
def metoda_elbow(ierarhie, nr_clusteri=None):
    n = ierarhie.shape[0] + 1
    if nr_clusteri is None:
        d = ierarhie[1:, 2] - ierarhie[:n - 2, 2]
        nr_jonctiuni = np.argmax(d) + 1
        nr_clusteri = n - nr_jonctiuni
    else:
        nr_jonctiuni = n - nr_clusteri
    threshold = (ierarhie[nr_jonctiuni, 2] + ierarhie[nr_jonctiuni - 1, 2]) / 2
    return nr_clusteri, threshold

# Calculare partitie ( In ce cluster apartine fiecare rand din x)
def calculare_partitie(k, x):
    ierarhie_clusteri = AgglomerativeClustering(k)
    clusteri = ierarhie_clusteri.fit_predict(x)
    partitie = np.array(["C" + str(cod+1) for cod in clusteri])
    return partitie


n, m = np.shape(set_date)
x = set_date[variabile].values

# print(x)

# Ierarhie:
metoda = "ward"
ierarhie = linkage(x, metoda)

# Determinare prime 2 componente pentru reprezentare partitii in axele acestora:
pca = PCA()
pca.fit(x)
componente = pca.transform(x)

# DataFrame ierarhie:
df_ierarhie = pd.DataFrame(ierarhie, columns=["Cluster 1", "Cluster 2", "Distanta", "Frecventa"])
# Asta seteaza numele coloanei aferenta indexului. Te doare mintea
df_ierarhie.index.name = "Jonctiune"
df_ierarhie.to_csv("data_out/Cluster/ierarhie.csv")

k_opt, threshold_opt = metoda_elbow(ierarhie)
print(k_opt, threshold_opt)

tabel_partitii = pd.DataFrame(index=set_date.index)

partitie_optima = calculare_partitie(k_opt, x)
# print(partitie_optima)

silhouette_opt = silhouette_samples(x, partitie_optima)

tabel_partitii["Partitie_Optima"] = partitie_optima
tabel_partitii["Silhouette_Partitie_Optima"] = silhouette_opt

# Calcul partitie din n clusteri (in cazul asta, 4):
k = 4
k, threshold_k = metoda_elbow(ierarhie=ierarhie, nr_clusteri=k)
partitie_k = calculare_partitie(k, x)

tabel_partitii["Partitie_" + str(k)] = partitie_k
tabel_partitii["Silhouette_Partitie_" + str(k)] = silhouette_samples(x, partitie_k)

tabel_partitii.to_csv("data_out/Cluster/partitii.csv")