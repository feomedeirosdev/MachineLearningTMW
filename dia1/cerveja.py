# %%
import pandas as pd

# %%
df = pd.read_excel('../datas/dados_cerveja.xlsx')
df

# %%
features = list(df.columns[1:-1])
target = df.columns[-1]

X = df[features]
y = df[target]

# %%
X = X.replace({
  'mud':0, 'pint':1,
  'não':0, 'sim': 1,
  'escura':0, 'clara':1
})
X
# %%
from sklearn import tree

# %%
arvore = tree.DecisionTreeClassifier()
arvore.fit(X, y)

# %%
tree.plot_tree(
  arvore,
  class_names=arvore.classes_,
  feature_names=features,
  filled=True
)
# %%
probas = arvore.predict_proba([[-1,1,1,1]])[0]
probas

# %%
pd.Series(probas, index=arvore.classes_)
# %%
