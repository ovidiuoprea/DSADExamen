import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from scipy.stats import f

from utils import nan_replace_t


def codificare(set_date, variabile_calitative):
    for v in variabile_calitative:
        if not is_numeric_dtype(set_date[v]):
            set_date[v] = pd.Categorical(set_date[v]).codes + 1

def calcul_indicatori_acuratete(y, y_, clase):
    matrice_confuzie = confusion_matrix(y, y_, labels=clase)
    acuratete = np.diag(matrice_confuzie) * 100 / np.sum(matrice_confuzie, axis=1)
    acuratete_medie = np.mean(acuratete)
    acuratete_globala = np.sum(np.diag(matrice_confuzie)) * 100 / len(y)
    scor_cohen_kappa = cohen_kappa_score(y, y_, labels=clase)

    df_matrice_confuzie = pd.DataFrame(matrice_confuzie, clase, clase)
    df_matrice_confuzie["Acuratete"] = acuratete

    indicatori_acuratete = pd.Series(
        [acuratete_globala, acuratete_medie, scor_cohen_kappa],
        ["Acuratete generala", "Acuratete medie", "Scor Cohen-Kappa"],
        name = "Acuratete")

    return df_matrice_confuzie, indicatori_acuratete

# Neaparat variabilele calitative trebuie citite cu :str
set_date = pd.read_csv("data_in/AD-ProiectB_train.csv", index_col=0, dtype={"RiscZidarie":str, "RiscStructura":str})
nan_replace_t(set_date)

predictori = list(set_date)[:-1]
tinta = list(set_date)[-1]

# RiscZidarie, RiscStructura sunt variabile calitative => Trebuie transformate
codificare(set_date, ["RiscStructura", "RiscZidarie"])

x_train, x_test, y_train, y_test = train_test_split(set_date[predictori], set_date[tinta], test_size=0.3)

model_lda = LinearDiscriminantAnalysis()
model_lda.fit(x_train, y_train)

# Putere de discriminare predictori:
clase = model_lda.classes_

n = len(x_train)
q = len(clase)

t = np.cov(x_train.values, rowvar=False)
probabilitati_clase = model_lda.priors_
g = model_lda.means_ - np.mean(x_train.values, axis=0)

b = g.T@np.diag(probabilitati_clase)@g
w = t - b

test_f_predictori = (np.diag(b) / (q-1))/(np.diag(w)/(n-q))
p_values = 1 - f.cdf(test_f_predictori, q-1, n-q)

predictori_ = list(np.array(predictori)[p_values < 0.1])

# Reconstruire model cu predictorii ramasi
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(x_train[predictori_],y_train)

m = len(predictori_)

# Calcul scoruri discriminante in model liniar:
scoruri = model_lda.transform(x_train[predictori_])
df_scoruri = pd.DataFrame(data=scoruri, index=x_train.index, columns=["Componenta" + str(i+1) for i in range(len(clase)-1)])

# Testarea modelului pe setul de antrenare:
y_test_lda = model_lda.predict(x_test[predictori_])
cm_lda, acc_lda = calcul_indicatori_acuratete(y_test, y_test_lda, clase)

cm_lda.to_csv("data_out/AD/matrice_confuzie_lda.csv")
acc_lda.to_csv("data_out/AD/acuratete_lda.csv")

# Predictie LDA pe setul de aplicare:
set_aplicare = pd.read_csv("data_in/AD-ProiectB_apply.csv", index_col=0, dtype={"RiscZidarie":str, "RiscStructura":str})
nan_replace_t(set_aplicare)
codificare(set_aplicare, ["RiscZidarie", "RiscStructura"])

tabel_predictii = pd.DataFrame(index=set_aplicare.index)

y_aplicare_lda = model_lda.predict(set_aplicare[predictori_])
tabel_predictii["Predictie LDA"] = y_aplicare_lda

print(tabel_predictii)

# Model Bayesian:
model_bayes = GaussianNB()
model_bayes.fit(x_train[predictori_], y_train)

y_test_bayes = model_bayes.predict(x_test[predictori_])
cm_bayes, acc_bayes = calcul_indicatori_acuratete(y_test, y_test_bayes, clase)

cm_lda.to_csv("data_out/AD/matrice_confuzie_bayes.csv")
acc_lda.to_csv("data_out/AD/acuratete_bayes.csv")

# Bayes pe model de aplicare:
y_aplicare_bayes = model_bayes.predict(set_aplicare[predictori_])

tabel_predictii["Predictie Bayes"] = y_aplicare_bayes

tabel_predictii.to_csv("data_out/AD/tabel_predictii.csv")