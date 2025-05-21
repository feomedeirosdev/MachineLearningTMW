# %%
import pandas as pd
from sklearn import tree

# %%
df = pd.read_excel('../datas/dados_frutas.xlsx')
df

# %%
features = list(df.columns)[:-1]
target = list(df.columns)[-1]

X = df[features]
y = df[target]

# %%

# Criando objeto decisionTreeClassifier
arvore = tree.DecisionTreeClassifier()

# Ensinando a máquina
arvore.fit(X, y)

# %%
tree.plot_tree(arvore,
               class_names = arvore.classes_,
               feature_names = features,
               filled = True)

# %%
# Arredondada	Suculenta	Vermelha	Doce
arvore.predict([[1,1,1,1]])

# %%
probas = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(probas, index=arvore.classes_)

# %%
