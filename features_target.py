from sklearn.model_selection import train_test_split
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins = penguins.dropna()
print(penguins.columns)
features = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']
penguins['is_adelie'] = penguins['species'].map(lambda s: 1 if s == "Adelie" else 0)
target = penguins['is_adelie']

x_train, x_val, y_train, y_val = train_test_split(penguins[features], target, test_size=0.2, random_state=20, stratify=target)

print(len(x_train), len(x_val))
