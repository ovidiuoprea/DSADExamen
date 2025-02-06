# ANALIZA CANONICA

import numpy as np
import pandas as pd
from factor_analyzer import calculate_bartlett_sphericity
from scipy.stats import chi2
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import normalize

from utils import nan_replace_t

def test_bartlett(c2, n, p, q, m):
    x = 1 - c2
    df = [(p-k+1) * (q - k + 1) for k in range(1, m+1)]
    l = np.flip(np.cumprod(np.flip(x)))
    chi2_ = (-n + 1 + (p + q + 1) / 2) * np.log(l)
    return 1 - chi2.cdf(chi2_, df)

tabel1 = pd.read_csv("data_in/AC-Y_DNA_Tari.csv", index_col=0)
nan_replace_t(tabel1)

tabel2 = pd.read_csv("data_in/AC-Mt_DNA_tari.csv", index_col=0)
nan_replace_t(tabel2)

variabile_set1 = list(tabel1)[1:]
variabile_set2 = list(tabel2)[1:]

p = len(variabile_set1)
q = len(variabile_set2)

# m = numarul maxim de perechi de variabile canonice obtinute
m = min(p,q)

# Neaparat trebuie merge pentru a se autostabiliza numarul de variabile (habar n-am)
tabel = pd.merge(tabel1[variabile_set1], tabel2[variabile_set2], left_index=True, right_index=True)
x = tabel[variabile_set1].values
y = tabel[variabile_set2].values
n = len(tabel)

# Construire model:
model_cca = CCA(n_components=m)
model_cca.fit(x, y)

# Scoruri canonice:
z, u = model_cca.transform(x, y)

# Normalizare scoruri canonice:
z = normalize(z, axis=0)
u = normalize(u, axis=0)

etichete_z = ["z" + str(i+1) for i in range (m)]
etichete_u = ["u" + str(i+1) for i in range (m)]

df_z = pd.DataFrame(z, tabel.index, etichete_z)
df_u = pd.DataFrame(u, tabel.index, etichete_u)

df_u.to_csv("data_out/AC/u.csv")
df_z.to_csv("data_out/AC/z.csv")

# Corelatii canonice:
corelatii = np.diag(np.corrcoef(z, u, rowvar=False)[:m, m:])
c2 = corelatii * corelatii

p_values = test_bartlett(c2, n, p, q, m)
# print(p_values)

etichete_radacini = ["root" + str(i+1) for i in range(m)]
df_semnificatie = pd.DataFrame(
    data={
        "Corelatii": corelatii,
        "Corelatii2": c2,
        "P-values": p_values
    }, index=etichete_radacini
)
df_semnificatie.to_csv("data_out/AC/root.csv")

# Numar radacini semnificative
numar_radacini_semnificative = np.where(p_values > 0.01)[0][0]

numar_radacini_semnificative = numar_radacini_semnificative + 1 if numar_radacini_semnificative == 1 else numar_radacini_semnificative
print("Radacini semnificative:", numar_radacini_semnificative)

# Analiza legatura:
corelatie_xz = np.corrcoef(x, z, rowvar=False)[:p, p:]
corelatie_yu = np.corrcoef(y, u, rowvar=False)[:q, q:]

df_corelatie_xz = pd.DataFrame(corelatie_xz, variabile_set1, etichete_z)
df_corelatie_yu = pd.DataFrame(corelatie_yu, variabile_set2, etichete_u)

df_corelatie_xz.to_csv("data_out/AC/xz.csv")
df_corelatie_yu.to_csv("data_out/AC/yu.csv")

# Varianta comuna:
vx = np.sum(corelatie_xz[:, :m] * corelatie_xz[:, :m], axis=0)
vy = np.sum(corelatie_yu[:, :m] * corelatie_yu[:, :m], axis=0)

# Varianta redundanta:
sx = vx * c2[:m]
sy = vy * c2[:m]

# DataFrame varianta / redundanta:
df_vr = pd.DataFrame(data ={
    'vx': vx, 'vy': vy, 'vx%': vx * 100 / p, 'vy%': vy * 100 / q,
    'sx': sx, 'sy': sy, 'sx%': sx * 100 / p, 'sy%': sy * 100 / q
}, index=etichete_radacini[:m])

df_vr.to_csv("data_out/AC/varianta_redundanta.csv")