import os
import pandas as pd
from sympy import symbols, Eq, solve

protein = ''
path = ''

#  Create folders for a target protein to remove redundant work
try:
    os.makedirs("./datasets/ga_interim_data/{}".format(protein))
    os.makedirs("./datasets/model_validation/{}".format(protein))
    os.makedirs("./datasets/training/{}".format(protein))
    os.makedirs("./model/{}".format(protein))
except FileExistsError:
    print("Directory exists, change folders name.")


#  Setup data for training and GA from score_sequences.csv
scored_sequences = pd.read_csv(path)
scored_sequences.to_csv("./datasets/ga_interim_data/{}".format(protein))

#  top_iter_0.csv from scored_sequences
top_iter_0 = scored_sequences.df.nlargest(200, 'Entropy')

import sympy as sp
x = sp.symbols('x', real=True, positive = True)

length = 1000 # len(scored_sequences.index)

expr = (15 * (length - 2*x) * (length-1 - 2*x) / 2 - 70 * x*(x-1)/2 )

sol = sp.solve(expr)[0]

print(sp.simplify(sol))


