import numpy as np
import pandas as pd
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo, FactorAnalyzer

set_date = pd.read_csv("../data_in/AF-Y_DNA_Tari.csv", index_col=0)
variabile = list(set_date)[1:]

x = set_date[variabile].values

test_bartlett = calculate_bartlett_sphericity(x)
test_kmo = calculate_kmo(x)

# Model FA
m,n = np.shape(x)

model_af = FactorAnalyzer(n_factors=m)
model_af.fit(x)

# Varianta factori:
varianta_factori = model_af.get_factor_variance()

# Corelatii factoriale:
corelatii_factoriale = model_af.loadings_

# Comunalitati, varianta specifica:
comunalitati = model_af.get_communalities()

varianta_specifica = model_af.get_uniquenesses()

# Scoruri:
scoruri = model_af.transform(x)