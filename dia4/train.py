# %%
import pandas as pd
df = pd.read_csv('../datas/dados_pontos.csv', sep=';')
df

# %%
list(df.columns)

# %%
features = list(df.columns[3:-1])
target = df.columns[-1]

X = df[features]
y = df[target]

# %%
from sklearn import model_selection

X_train, X_test, y_train, y_test = (
  model_selection.train_test_split(
  X, 
  y, 
  test_size = 0.2,
  random_state = 42,
  stratify = y
))

# %%
print(f'Tx resposta y treino: {y_train.mean()*100:.2f}%')
print(f'Tx resposta y test: {y_test.mean()*100:.2f}%')

# %%
