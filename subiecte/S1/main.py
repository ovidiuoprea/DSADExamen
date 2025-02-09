import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split

salarii = pd.read_csv("data_in/E_NSAL_2008-2021.csv", index_col=0)
variabile_salarii = list(salarii)
populatie_localitati = pd.read_csv("data_in/PopulatieLocalitati.csv", index_col=0)

# A
# 1. # Anul cu cei mai multi angajati pentru fiecare localitate:

def f_cerinta1(x):
    index_an_maxim = np.argmax(x)
    an_maxim = variabile_salarii[index_an_maxim]
    return pd.Series(variabile_salarii[index_an_maxim], ["Anul"])

cerinta1 = salarii.apply(func = lambda x: f_cerinta1(x), axis=1)
cerinta1.to_csv("data_out/Cerinta1.csv")

# 2. Rata ocuparii populatiei pe an si rata medie la nivel de judet, in ordine descrescatoare

def f_rata_medie(x):
    medie = np.mean(x, axis=0)
    return pd.Series(medie)

def f_rata_ocupare(x):
    rata_ocupare = x[variabile_salarii] / x["Populatie"]
    return pd.Series(rata_ocupare)

salarii_populatie = pd.merge(salarii[variabile_salarii], populatie_localitati["Populatie"], left_index=True, right_index=True)

rata_ocupare_localitate = salarii_populatie.apply(func = lambda x: f_rata_ocupare(x), axis=1)
salarii_populatie_judet = pd.merge(salarii_populatie, populatie_localitati["Judet"], left_index=True, right_index=True)


sum_salarii_judet = salarii_populatie_judet.groupby(by="Judet").sum()

rata_ocupare_judet = sum_salarii_judet.apply(func=lambda x: f_rata_ocupare(x), axis=1)
rata_ocupare_judet["RataMedie"] = rata_ocupare_judet.apply(func = lambda x: f_rata_medie(x), axis=1)

rata_ocupare_judet.to_csv("data_out/Cerinta2.csv")

# B. Clasificare analiza liniara discriminanta:
set_date = pd.read_csv("data_in/Pacienti.csv", index_col=0)
predictori = list(set_date)[:-1]
tinta = list(set_date)[-1]

# Inlocuire valori NAN
def f_nan_replace_t(t: pd.DataFrame):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if is_numeric_dtype(t[v]):
            t.fillna({v: t[v].mean()}, inplace=True)
        else:
            t.fillna({v: t[v].mode()[0]}, inplace=True)

f_nan_replace_t(set_date)

# Impartire in train / test
x_train, x_test, y_train, y_test = train_test_split(set_date[predictori], set_date[tinta], test_size=0.3)

# Aplicare analiza liniara discriminanta:
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(x_train, y_train)

# Scorurile discriminante:
z = model_lda.transform(x_test[predictori])

clase = model_lda.classes_

df_z = pd.DataFrame(z, x_test.index, ["C"+str(i+1) for i in range(z.shape[1])])
df_z.to_csv("data_out/z.csv")

# Graficul scorurilor discriminante in primele 2 axe discriminante:

def plot_scoruri(z):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Grafic scoruri in primele 2 axe discriminante")
    ax.set_xlabel("Axa 1")
    ax.set_ylabel("Axa 2")
    ax.scatter(z[:, 0], z[: ,1])
def show():
    plt.show()

plot_scoruri(z)
# show()

# Performante model - pe setul de antrenare

def calculare_metrici(y_real, y_prezis, clase):
    matrice_confuzie = confusion_matrix(y_real, y_prezis)
    acuratete = np.diag(matrice_confuzie) * 100 / np.sum(matrice_confuzie, axis=1)

    df_confuzie = pd.DataFrame(matrice_confuzie)
    # print(df_confuzie)
    df_confuzie["Acuratete"] = acuratete
    df_confuzie.to_csv("data_out/mac.csv")

    # indicatori acuratete:
    acuratete_medie = np.mean(acuratete)
    acuratete_globala = np.sum(np.diag(matrice_confuzie)) * 100 / len(y_real)
    index_ck = cohen_kappa_score(y_real, y_prezis)

    print("Acuratete medie: ", acuratete_medie)
    print("Acuratete globala: ", acuratete_globala)
    print("Indice Cohen-Kappa: ", index_ck)

predictie_antrenare = model_lda.predict(x_test)

calculare_metrici(y_real=y_test, y_prezis=predictie_antrenare, clase=clase)
