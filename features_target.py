import pandas as pd
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins = penguins.dropna()
print(penguins.columns)
features = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']
target = 'species'