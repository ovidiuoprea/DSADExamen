# ANALIZA FACTORIALA

import pandas as pd
import numpy as np
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo, FactorAnalyzer

set_date = pd.read_csv("data_in/AF-Y_DNA_Tari.csv", index_col=0)
variabile = list(set_date)[1:]

x = set_date[variabile].values

# 1. Calculare test Bartlett:
test_bartlett = calculate_bartlett_sphericity(x)
df_test_bartlett = pd.DataFrame([test_bartlett[0], test_bartlett[1]], ["Valoare", "P-value"], ["Test Bartlett"])
df_test_bartlett.to_csv("data_out/AF/bartlett.csv")

if test_bartlett[1] > 0.01:
    print("Model nefactorizabil!")

# 2. KMO
kmo = calculate_kmo(x)

df_kmo = pd.DataFrame({
    "Index KMO": np.append(kmo[0], kmo[1])
}, [variabile + ["Index global"]])
df_kmo.to_csv("data_out/AF/kmo.csv")

# Creare model
n, m = np.shape(x)

model_fact = FactorAnalyzer(m, rotation="varimax")
# model_fact = FactorAnalyzer(m, rotation="none")
model_fact.fit(x)

# Varianta factori:
varianta_factori = model_fact.get_factor_variance()

df_varianta_factori = pd.DataFrame(
    {
        "Varianta": varianta_factori[0],
        "Procent varianta": varianta_factori[1] * 100,
        "Procent cumulat": varianta_factori[2] * 100
    },
    ["F" + str(i+1) for i in range(m)]
)
df_varianta_factori.to_csv("data_out/AF/varianta_factori.csv")

# Analiza corelatiilor:
corelatii = model_fact.loadings_
df_corelatii = pd.DataFrame(corelatii, variabile, ["F" + str(i+1) for i in range(m)])
df_corelatii.to_csv("data_out/AF/corelatii.csv")

# Scoruri:
scoruri = model_fact.transform(x)

df_scoruri = pd.DataFrame(scoruri, set_date.index, ["F" + str(i+1) for i in range(m)])
df_scoruri.to_csv("data_out/AF/scoruri.csv")

#Varianta specifica:
varianta_specifica = model_fact.get_uniquenesses()
df_varianta_specifica = pd.DataFrame(
    {
        "Varianta": varianta_specifica
    }, variabile
)
df_varianta_specifica.to_csv("data_out/AF/varianta_specifica.csv")

#Comunalitati
comunalitati = model_fact.get_communalities()
df_comunalitati = pd.DataFrame({
    "Comunalitate": comunalitati
}, variabile)
df_comunalitati.to_csv("data_out/AF/comunalitati.csv")

