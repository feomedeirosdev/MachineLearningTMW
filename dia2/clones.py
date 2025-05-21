# %%
import pandas as pd
df = pd.read_parquet('../datas/dados_clones.parquet')
df.columns = list(df.columns.str.strip())

# %%
df_estatura_massa = (
  df
  .groupby(['Status'], as_index=False)
  .agg(
    Estatura_cm=('Estatura(cm)', 'mean'),
    Massa_kg=('Massa(em kilos)', 'mean'),
  )
)

diff_estatura = (
  df_estatura_massa ['Estatura_cm'].iloc[0] - 
  df_estatura_massa ['Estatura_cm'].iloc[1]
)

diff_massa = (
  df_estatura_massa ['Massa_kg'].iloc[0] - 
  df_estatura_massa ['Massa_kg'].iloc[1]
)

linha_diff = pd.DataFrame({
    'Status': ['Diff'],
    'Estatura_cm': [abs(diff_estatura)],
    'Massa_kg': [abs(diff_massa)]
})

df_estatura_massa = pd.concat(
  [df_estatura_massa, linha_diff], 
  ignore_index=True
)

df['Status_bool'] = df['Status'] == 'Apto'

df_dist_ombro = (
  df
  .groupby('Distância Ombro a ombro', as_index=False)
  .agg(
    Status_bool = ('Status_bool', 'mean')
  )
)

df_tam_cranio = (
  df
  .groupby('Tamanho do crânio', as_index=False)
  .agg(
    Status_bool = ('Status_bool', 'mean')
  )
)

df_tam_pe = (
  df
  .groupby('Tamanho dos pés', as_index=False)
  .agg(
    Status_bool = ('Status_bool', 'mean')
  )
)

df_gen_encarregado = (
  df
  .groupby('General Jedi encarregado', as_index=False)
  .agg(
    Status_bool = ('Status_bool', 'mean')
  )
)

# %%
df_estatura_massa

# %%
df_dist_ombro

# %%
df_tam_cranio

# %%
df_tam_pe

# %%
df_gen_encarregado

# %%
features = [
  'Massa(em kilos)',
  'Estatura(cm)',
  'Distância Ombro a ombro',
  'Tamanho do crânio',
  'Tamanho dos pés',
]

cat_features = [
  'Distância Ombro a ombro',
  'Tamanho do crânio',
  'Tamanho dos pés',
]

target = 'Status'

X = df[features]
y = df[target]

X

# %%
from feature_engine import encoding

onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(X)

X = onehot.transform(X)
X

# %%
x_columns = list(X.columns)
x_columns
# %%
from sklearn import tree

# %%
arvore = tree.DecisionTreeClassifier(max_depth=2)
arvore.fit(X, df['Status'])

# %%
import matplotlib.pyplot as plt
plt.figure(dpi=600)
tree.plot_tree(
  arvore,
  class_names=arvore.classes_,
  feature_names=x_columns, 
  filled=True,
  
)
# %%
