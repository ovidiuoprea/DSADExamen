import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity


def f_cerinta1(x):
    unde = np.argmin(x[variabile_vot])
    return pd.Series([x["Localitate"], variabile_vot[unde]], ["Localitate","Categorie"])

vot = pd.read_csv("data_in/vot_generat.csv", index_col=0)
coduri_localitati = pd.read_csv("data_in/coduri_localitati_generat.csv", index_col=0)
variabile = list(vot)
variabile_vot = list(vot)[1:]
# print(variabile_vot)

cerinta1 = vot.apply(func = lambda x: f_cerinta1(x), axis=1)
cerinta1.to_csv("data_out/cerinta1.csv")

vot_judet = pd.merge(vot[variabile_vot], coduri_localitati["Judet"], left_index=True, right_index=True)

cerinta2 = vot_judet.groupby(by="Judet").mean()
cerinta2.to_csv("data_out/cerinta2.csv")

# Analiza factoriala fara rotatie de factori

x = vot[variabile_vot].values

model_fact = FactorAnalyzer(rotation=None)
model_fact.fit(x)

n, m = np.shape(x)

test_bartlett = calculate_bartlett_sphericity(x)

print("Valoare test Bartlett: ", test_bartlett[0])
print("P-value asociat test Bartlett: ", test_bartlett[1])

scoruri = model_fact.transform(x)
df_scoruri = pd.DataFrame(scoruri, vot.index, ["F" + str(i+1) for i in range(scoruri.shape[1])])

print(df_scoruri)

