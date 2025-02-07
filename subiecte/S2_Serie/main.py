import numpy as np
import pandas as pd

def f_cerinta1(x):
    medie = np.mean(x[variabile_ani])
    return pd.Series([x["Localitate"], medie], ["Localitate", "Diversitate"])

def f_cerinta2(x):
    vector_localitati_zero = []
    for an in variabile_ani:
        numar_localitati_nonzero = np.count_nonzero(x[an])
        numar_localitati_zero = len(x[an]) - numar_localitati_nonzero
        vector_localitati_zero.append(numar_localitati_zero)
    return pd.Series(vector_localitati_zero, variabile_ani)

set_date = pd.read_csv("data_in/Diversitate.csv", index_col=0)
coduri_judete = pd.read_csv("data_in/coduri_localitati.csv", index_col=0)

variabile = list(set_date)
# variabile_ani = variabilele numerice din setul de date = anii => ignor coloana 1 (Localitate)
variabile_ani = list(set_date)[1:]

# A.
# Cerinta 1.

cerinta1 = set_date.apply(func = lambda x: f_cerinta1(x), axis=1)
# print(cerinta1)
cerinta1_sortata = cerinta1.sort_values(by="Diversitate", ascending=False)
# print(cerinta1_sortata)
cerinta1_sortata.to_csv("data_out/cerinta1.csv")

merge_diversitate_judet = pd.merge(set_date, coduri_judete["Judet"], left_index=True, right_index=True)

cerinta2 = merge_diversitate_judet.groupby(by="Judet").apply(func = lambda x: f_cerinta2(x), include_groups = False)
cerinta2.to_csv("data_out/cerinta2.csv")