# ANALIZA IN COMPONENTE PRINCIPALE

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def tabelare_varianta(alpha):
    procent_alpha = alpha * 100 / sum(alpha)
    return pd.DataFrame(data={
        "Varianta": alpha,
        "Varianta cumulata": np.cumsum(alpha),
        "Procent varianta": procent_alpha,
        "Procent varianta cumulata": np.cumsum(procent_alpha)
    }, index=["C" + str(i+1) for i in range(m)])

def cerc_corelatii(t: pd.DataFrame, corelatii=True):
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(1,1,1)
    if corelatii:
        tetha = np.arange(0, np.pi * 2, 0.01)
        ax.plot(np.sin(tetha), np.cos(tetha))
        ax.plot(0.7 * np.sin(tetha), 0.7 * np.cos(tetha))
    ax.scatter(t[0], t[1], c="r")

def show():
    plt.show()

tabel_date = pd.read_csv("data_in/ACP-Ethnicity.csv", index_col=0)
variabile = list(tabel_date)[1:]

x = tabel_date[variabile].values


# Standardizare date:
standardizare = StandardScaler()
x = standardizare.fit_transform(x)

# Construire model:
model_acp = PCA()
model_acp.fit(x)

n, m = np.shape(x)

# Varianta in componente:
alpha = model_acp.explained_variance_ * (n-1) / n
df_alpha = tabelare_varianta(alpha)

df_alpha.to_csv("data_out/ACP/varianta.csv")

# Scoruri:
componente = model_acp.transform(x)

scoruri = componente / np.sqrt(alpha)

df_scoruri = pd.DataFrame(scoruri, tabel_date.index, df_alpha.index)
df_scoruri.to_csv("data_out/ACP/scoruri.csv")

# Corelatii factoriale:
corelatii = np.corrcoef(x, componente, rowvar=False)[:len(alpha), len(alpha):]
cerc_corelatii(corelatii, True)
show()


# Cosinusuri:
c2 = corelatii * corelatii
cosinusuri = (c2.T / np.sum(c2, axis=1)).T

df_cosinusuri = pd.DataFrame(data=cosinusuri, index=variabile, columns=["C" + str(i+1) for i in range(m)])
df_cosinusuri.to_csv("data_out/ACP/cosinusuri.csv")

# Comunalitati:
comunalitati = np.cumsum(corelatii * corelatii, axis=1)

df_comunalitati = pd.DataFrame(data=comunalitati, index=variabile, columns=df_alpha.index)
df_comunalitati.to_csv("data_out/ACP/comunalitati.csv")

# Contributii:
contributii = c2 / np.sum(c2, axis=1)
df_contributii = pd.DataFrame(data=contributii, index=variabile, columns=df_alpha.index)
df_contributii.to_csv("data_out/ACP/contributii.csv")





# print(tabel_date)
def f_cerinta(x: pd.DataFrame):
    cate0 = len(np.where(x[variabile] > 55000)[0])
    if cate0 >= 1:
        return x

cerinta = tabel_date.apply(f_cerinta, axis=1)
# print(cerinta)

tabel_nou = tabel_date.copy()
tabel_nou["Contor"] = tabel_date.apply(func=lambda x: len(np.where(x[variabile] > 55000)[0]), axis=1)
test_cerinta = tabel_nou[tabel_nou["Contor"] > 0]
# print(test_cerinta[["City"] +  variabile])

judete = pd.read_csv("data_in/Coduri_Localitati.csv", index_col=0)
# print(judete)

date_judet_merged = pd.merge(tabel_date, judete["County"], left_index=True, right_index=True)
# print(date_judet_merged)

def f_cerinta2examen(x: pd.DataFrame):
    # print(x.columns)
    contoare = []


    for coloana in x.columns:
        # print(coloana, x[coloana], sep="\n")
        date_an = x[coloana]
        cate0 = len(np.where(x[coloana] == 0)[0])
        contoare.append(cate0)

    return pd.Series(contoare, x.columns)

cerinta2_examen = date_judet_merged.groupby(by="County").apply(f_cerinta2examen, include_groups=False)

# print(cerinta2_examen)

def functie5(x: pd.DataFrame):
    return (x == 0).sum(axis=0)


cerinta5 = tabel_date[variabile].merge(judete["County"], left_index=True, right_index=True)
cerinta5 = cerinta5.groupby("County").apply(func=functie5, include_groups=False)

# print(cerinta2_examen - cerinta5)