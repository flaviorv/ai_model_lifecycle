from palmerpenguins import load_penguins
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

penguins = load_penguins()
penguins = penguins.dropna()

cont_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year']

penguins['sex_bin'] = penguins['sex'].map({'male':  0, 'female': 1})
island_dummies = pd.get_dummies(penguins['island'], prefix='island')
island_dummies = island_dummies.astype(int)
penguins = pd.concat([penguins, island_dummies], axis=1)
cat_features = ['sex_bin'] + list(island_dummies.columns)

features = cat_features + cont_features

penguins['is_adelie'] = penguins['species'].map(lambda s: 1 if s == "Gentoo" else 0)
target = penguins['is_adelie']

x_train, x_val, y_train, y_val = train_test_split(penguins[features], target, test_size=0.2, random_state=20, stratify=target)

scaler = MinMaxScaler()
x_train[cont_features] = scaler.fit_transform(x_train[cont_features])
x_val[cont_features] = scaler.transform(x_val[cont_features])

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
pred = knn.predict(x_val)

print(classification_report(y_val, pred, target_names=['Others', 'Gentoo']))