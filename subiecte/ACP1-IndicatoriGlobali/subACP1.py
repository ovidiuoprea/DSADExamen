import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def grafic_scoruri(scoruri):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Grafic scoruri in primele 2 axe")
    ax.set_xlabel("C1")
    ax.set_ylabel("C2")
    ax.scatter(scoruri[:, 0], scoruri[:, 1])

def show():
    plt.show()

global_indicators = pd.read_csv("data_in/GlobalIndicatorsPerCapita_2021.csv", index_col=0)
v_global_indicators = list(global_indicators)
v_numerice = v_global_indicators[1:]

valoare_adaugata = global_indicators.apply(func = lambda x: pd.Series([x["Country"], np.sum(x[v_numerice])], ["Country", "Valoare adaugata"]), axis=1)
valoare_adaugata.to_csv("data_out/cerinta1.csv")

country_continents = pd.read_csv("data_in/CountryContinents.csv", index_col=0)
gi_continente = pd.merge(global_indicators, country_continents["Continent"], left_index=True, right_index=True)

cerinta2 = gi_continente.groupby(by="Continent").apply(func=lambda x: pd.Series(data=(np.std(x[v_numerice], axis=0) / np.mean(x[v_numerice], axis=0)).values, index=v_numerice), include_groups=False)
cerinta2.to_csv("data_out/cerinta2.csv")

# B. Analiza in componente principale:

x = global_indicators[v_numerice].values
standardizare = StandardScaler()
x = standardizare.fit_transform(x)

model_acp = PCA()
model_acp.fit(x)

# 1. Variantele componentelor principale:
n, m = np.shape(x)
alpha = model_acp.explained_variance_ratio_ * (n-1) / n

etichete_c = ["C"+str(i+1) for i in range(m)]

df_alpha = pd.DataFrame(data=alpha, index= etichete_c, columns=["Varianta"])
df_alpha.to_csv("data_out/varianta.csv")

# 2. Scoruri:
componente = model_acp.transform(x)

scoruri = componente / np.sqrt(alpha)
print(np.shape(scoruri), n, m, sep="\n")

df_scoruri = pd.DataFrame(scoruri, global_indicators.index, etichete_c)
df_scoruri.to_csv("data_out/scoruri.csv")

# 3. Graficul scorurilor:
grafic_scoruri(scoruri)
# show()

# 4. Varianta specifica:
factor_loadings = pd.read_csv("data_in/g20.csv", index_col=0)

# Varianta specifica = 1 - comunalitate

# Comunalitate = suma(coeficienti_factoriali ^ 2)

communalities = np.cumsum(factor_loadings * factor_loadings, axis=1)
print(communalities.sum().idxmax())