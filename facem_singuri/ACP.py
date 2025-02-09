import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

tabel_date = pd.read_csv("../data_in/ACP-Ethnicity.csv", index_col=0)
variabile = list(tabel_date)[1:]

x = tabel_date[variabile].values

standardizare = StandardScaler()
x = standardizare.fit_transform(x)

model_acp = PCA()
model_acp.fit(x)

n, m = np.shape(x)
alpha = model_acp.explained_variance_ * (n-1) / n

componente = model_acp.transform(x)

corelatii = np.corrcoef(x, componente, rowvar=False)[:m, m:]
c2 = corelatii * corelatii
cosinusuri = (c2.T / np.sum(c2, axis=1)).T
contributii = c2 / np.sum(c2, axis=1)
comunalitati = np.cumsum(corelatii * corelatii)

print(cosinusuri)

def plot_varianta_componente(alpha):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Plot varianta componente")
    x = np.arange(1, len(alpha)+1)
    ax.set_xticks(x)
    ax.plot(x, alpha)

def

def show():
    plt.show()

plot_varianta_componente(alpha)
show()