import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

set_date = pd.read_csv("data_in/GlobalIndicatorsPerCapita_2021.csv", index_col=0)
coduri_tari = pd.read_csv("data_in/CoduriTari.csv", index_col=0)

indicatori = list(set_date)[1:]
indicatori_valoare_adaugata = ['AgrHuntForFish', 'Construction', 'Manufacturing', 'MiningManUt', 'TradeT', 'TransportComm', 'Other']

# A.
# 1. Valoarea adaugata pentru fiecare tara (suma indicatori valoare adaugata pe ramuri)

cerinta1 = set_date.apply(func=lambda x: pd.Series([x["Country"], np.sum(x[indicatori_valoare_adaugata], axis=0)], ["Country", "Valoare adaugata"]), axis=1)
cerinta1.to_csv("data_out/Cerinta1.csv")

def f_cerinta2(x):
    cv = np.std(x, axis=0) / np.mean(x, axis=0)
    return pd.Series(cv)

# Coeficient de variatie = abatere standard / medie
set_date_continent = pd.merge(set_date[indicatori], coduri_tari["Continent"], left_index=True, right_index=True)
cv_continente = set_date_continent.groupby(by="Continent").apply(func = lambda x: f_cerinta2(x), include_groups=False)

cv_continente.to_csv("data_out/Cerinta2.csv")

# B. Analiza in componente principale:

x = set_date[indicatori].values
standardizare = StandardScaler()
x = standardizare.fit_transform(x)

# Model ACP
model_acp = PCA()
model_acp.fit(x)
n, m = np.shape(x)

# Variante componente principale:
alpha = model_acp.explained_variance_ * (n-1) / n
print(" Variantele componentelor principale: ", alpha)

# Scorurile asociate instantelor:
componente = model_acp.transform(x)
scoruri = componente / np.sqrt(alpha)

corelatii = np.corrcoef(x, componente, rowvar=False)[:len(alpha), len(alpha):]
c2 = corelatii * corelatii

cos = (c2.T / np.sum(c2, axis=1)).T
contrib = c2 / np.sum(c2, axis=1)


comun = np.cumsum(contrib * contrib, axis=1)


df_scoruri = pd.DataFrame(scoruri, index=set_date.index, columns=["C" + str(i+1) for i in range(m)])
df_scoruri.to_csv("data_out/scoruri.csv")

# Am vrut sa vad daca mai tin minte graficul variantei, nu intra in cerinte
def plot_varianta(alpha, prag_minim):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Plot varianta")
    x = np.arange(1, len(alpha)+1)
    ax.set_xticks(x)
    ax.axhline(prag_minim, c="orange")
    kattel = np.argmin(alpha[alpha > 1])
    ax.axhline(kattel, c="r")
    ax.plot(x, alpha)


# Graficul scorurilor in primele 2 axe principale:
def grafic_scoruri(scoruri):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Grafic scoruri in primele 2 axe")
    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")
    ax.scatter(scoruri[:, 0], scoruri[:, 1])

def show():
    plt.show()

plot_varianta(alpha, 0.8)
grafic_scoruri(scoruri)
show()