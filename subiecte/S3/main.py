import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import normalize

electricity_production = pd.read_csv("data_in/ElectricityProduction.csv", index_col=0)
emissions = pd.read_csv("data_in/Emissions.csv", index_col=0)
populatie_europa = pd.read_csv("data_in/PopulatieEuropa.csv", index_col=0)

variabile_emissions = list(emissions)[1:]

emissions_tone = emissions.copy()
emissions_tone["GreenGE"] = emissions_tone["GreenGE"] * 1000
emissions_tone["GreenGIE"] = emissions_tone["GreenGIE"] * 1000

# A.

# Cerinta 1. Emisiile totale de particule, la nivel de tara, exprimate in tone
def f_cerinta1(x):
    return pd.Series([x["Country"], np.sum(x[variabile_emissions], axis=0)], ["Country", "Emisii_total_tone"])
cerinta1 = emissions_tone.apply(func = lambda x: f_cerinta1(x), axis=1)
cerinta1.to_csv("data_out/Cerinta1.csv")

# Cerinta 2. Emisiile la 100000 de locuitori, la nivel de regiune, pe tip de particule (tone / 100000loc)

# Sper ca e bine asa, nu inteleg prea bine ce vrea de la mine:
# populatie .... x tone
# 10000 loc .... y => y = x * 100000 / populatie

emissions_tone_regiune = pd.merge(emissions_tone[variabile_emissions], populatie_europa["Population"], left_index=True, right_index=True)
emissions_tone_regiune = pd.merge(emissions_tone_regiune, populatie_europa["Region"], left_index=True, right_index=True)

def f_cerinta2(x):
    emisii_particule = x[variabile_emissions] * 100000 / x["Population"]
    return pd.Series(emisii_particule)

emissions_grupate_regiune = emissions_tone_regiune.groupby(by="Region").sum()

cerinta2 = emissions_grupate_regiune.apply(func = lambda x: f_cerinta2(x), axis=1)
cerinta2.to_csv("data_out/Cerinta2.csv")

# Cerinta B.
variabile_set1 = list(electricity_production)[1:]
variabile_set2 = list(emissions)[1:]

tabel_date = pd.merge(electricity_production[variabile_set1], emissions[variabile_set2], left_index=True, right_index=True)

x = tabel_date[variabile_set1].values
y = tabel_date[variabile_set2].values

p = len(variabile_set1)
q = len(variabile_set2)

m = min(p, q)
n = len(tabel_date)

# Modelul canonic:
model_canonic = CCA(n_components=m)
model_canonic.fit(x, y)

# Scorurile canonice:
z, u = model_canonic.transform(x,y)

# Normalizare scoruri canonice:
z = normalize(z)
u = normalize(u)

etichete_z = ["Z" + str(i+1) for i in range(m)]
etichete_u = ["U" + str(i+1) for i in range(m)]

df_z = pd.DataFrame(z, tabel_date.index, etichete_z)
df_u = pd.DataFrame(u ,tabel_date.index, etichete_u)

df_z.to_csv("data_out/z.csv")
df_u.to_csv("data_out/u.csv")

# Corelatii canonice:
r = np.diag(np.corrcoef(z, u, rowvar=False)[:m, m:])

# Aplicare test Bartlett pentru verificare semnificatie statistica perechi canonice:
r2 = r * r

def test_bartlett(c2, n, p, q, m):
    x = 1 - c2
    df = [(p - k +1) * (q-k+1) for k in range(1, m+1)]
    l = np.flip(np.cumprod(np.flip(x)))
    chi2_ = (-n + 1 + (p+q+1) / 2) * np.log(l)
    return 1 - chi2.cdf(chi2_, df)

radacini_semnificative = test_bartlett(r2, n, p, q, m)
print("p-values test bartlett: ", radacini_semnificative)

numar_radacini_semnificative = np.where(radacini_semnificative > 0.01)[0][0]
numar_radacini_semnificative = numar_radacini_semnificative + 1 if numar_radacini_semnificative == 1 else numar_radacini_semnificative
print("Numar radacini semnificative: ", numar_radacini_semnificative)
