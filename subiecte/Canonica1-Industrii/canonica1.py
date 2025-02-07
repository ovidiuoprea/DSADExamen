import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


def f_cerinta1(x):
    medie = x[v_industrie] / x["Populatie"]
    return pd.Series(np.append(x["Localitate"], np.array(medie.values)), [np.append("Localitate", v_industrie)])

def f_cerinta2(x):
    unde = np.argmax(x)
    denumire = v_industrie[unde]
    valoare = x.iloc[unde]
    return pd.Series([denumire, valoare],["Activitate Dominanta", "Cifra Afaceri"])

def biplot(x, y):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Plot instante")
    ax.set_xlabel("Z1/U1")
    ax.set_ylabel("Z2/U2")
    # Componentele Z1, Z2; U1, U2 sunt echivalente cu indicii 0 si 1 in python!
    ax.scatter(x[:, 0], x[:, 1], c="r", label="X")
    ax.scatter(y[:, 0], y[:, 1], c="b", label="Y")
    ax.legend()

def show():
    plt.show()

industrie = pd.read_csv("data_in/Industrie.csv", index_col=0)
v_industrie = list(industrie)[1:]

populatie_localitati = pd.read_csv("data_in/PopulatieLocalitati.csv", index_col=0)

industrie_populatie = pd.merge(industrie, populatie_localitati["Populatie"], left_index=True, right_index=True)

cerinta1 = industrie_populatie.apply(func=lambda x: f_cerinta1(x), axis=1)
cerinta1.to_csv("data_out/cerinta1.csv")

merge_industrie_judet = pd.merge(industrie, populatie_localitati["Judet"], left_index=True, right_index=True)
industrie_judet = merge_industrie_judet[["Judet"] + v_industrie].groupby(by="Judet").sum()

cerinta2 = industrie_judet.apply(func=lambda x: f_cerinta2(x), axis=1)
cerinta2.to_csv("data_out/cerinta2.csv")

# Analiza canonica:
tabel_date = pd.read_csv("data_in/DataSet_34.csv", index_col=0)
x = tabel_date.values

standardizare = StandardScaler()
x = standardizare.fit_transform(x)

tabel_standardizat = pd.DataFrame(x, tabel_date.index, tabel_date.columns)

print(tabel_standardizat.columns)

variabile_set1 = ['prodPorc', 'prodVite', 'prodOaieSiCapra', 'prodPasareDeCurte']
variabile_set2 = ['consPorc', 'consVita', 'consumOaieSiCapra', 'consPasareDeCurte']

p = len(variabile_set1)
q = len(variabile_set2)

m = min(p,q)

x = tabel_standardizat[variabile_set1].values
y = tabel_standardizat[variabile_set2].values

model_cca = CCA(n_components=m)
model_cca.fit(x, y)

z, u = model_cca.transform(x, y)
print(np.shape(x))
print(np.shape(z))

etichete_z = ["Z" + str(i+1) for i in range(m)]
etichete_u = ["U" + str(i+1) for i in range(m)]

df_z = pd.DataFrame(data=z, index=tabel_standardizat.index, columns=etichete_z)
df_u = pd.DataFrame(data=u, index=tabel_standardizat.index, columns=etichete_u)

df_z.to_csv("data_out/Xscore.csv")
df_u.to_csv("data_out/Yscore.csv")

# Factor loadings:
r_xz = np.corrcoef(x, z, rowvar=False)[:p, p:]
r_yu = np.corrcoef(y, u, rowvar=False)[:q, q:]

print(np.shape(r_xz))

df_rxz = pd.DataFrame(r_xz, variabile_set1, etichete_z)
df_ryu = pd.DataFrame(r_yu, variabile_set2, etichete_u)

df_rxz.to_csv("data_out/Rxz.csv")
df_ryu.to_csv("data_out/Ryu.csv")

biplot(z, u)
show()