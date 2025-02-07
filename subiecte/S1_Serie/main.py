import numpy as np
import pandas as pd

def f_cerinta1(x):
    # Ma intereseaza sa vad cati din indicii pe an sunt 0.
    # np.count_nonzero numara tot ce e diferit de 0,
    # deci inversul este numarul de ani - cate variabile sunt diferite de 0
    cate_variabile_non_zero = np.count_nonzero(x[variabile_ani])
    cate_variabile_zero = len(variabile_ani) - cate_variabile_non_zero
    return pd.Series([cate_variabile_zero], ["Count_zero"])

def f_cerinta2(x):
    print(x)
    # Calculez mediile pe randuri pentru fiecare localitate
    diversitate_medie = np.mean(x[variabile_ani], axis=1)
    # Aflu media cea mai mare si pe ce pozitie se afla
    medie_maxima = np.max(diversitate_medie)
    pozitie_diversitate_maxima = np.argmax(diversitate_medie)
    # Iau din coloana "Localitate" numele localitatii aferente pozitiei unde media e max
    localitate = x["Localitate"].iloc[pozitie_diversitate_maxima]

    return pd.Series([localitate, medie_maxima], ["Localitate", "Diversitate maxima"])

set_date = pd.read_csv("data_in/Diversitate.csv", index_col=0)

variabile = list(set_date)
# variabile_ani = variabilele numerice din setul de date = anii => ignor coloana 1 (Localitate)
variabile_ani = list(set_date)[1:]

# A.
# Cerinta 1.
# Adaug o coloana noua in care am calculat cate dintre variabile sunt 0 pentru fiecare rand:
set_date_zero = set_date.copy()
set_date_zero["Zero"] = set_date.apply(func = lambda x: f_cerinta1(x), axis=1)
# print(set_date_zero)

# Filtrez setul de date dupa coloana "Zero" > 0 ( Pastrez doar randurile care au cel putin un zero)
set_date_zero = set_date_zero[set_date_zero["Zero"] > 0]
# print(set_date_zero)

# Pentru ca in output vrea doar Siruta, Oras si anii, salvez in csv fara coloana "Zero":
set_date_zero[variabile].to_csv("data_out/cerinta1.csv")

# Cerinta 2.
coduri_localitati = pd.read_csv("data_in/coduri_localitati.csv", index_col=0)
merge_diversitate_judet = pd.merge(set_date, coduri_localitati["Judet"], left_index=True, right_index=True)

cerinta2 = merge_diversitate_judet.groupby(by="Judet").apply(func=lambda x: f_cerinta2(x), include_groups=False)
cerinta2.to_csv("data_out/cerinta2.csv")