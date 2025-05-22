# %%
import pandas as pd
df = pd.read_excel('../datas/dados_cerveja_nota.xlsx')
df

# %%
df['aprovado'] = df['nota'] >= 5
df

# %%
features = ['cerveja']
target = 'aprovado'

X = df[features]
y_obs = df[target] # observado

# %%
from sklearn import linear_model
reg = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)

reg.fit(X, y)
y_reg_predict = reg.predict(X)
y_reg_predict # predição da regressão logística

# %%
from sklearn import metrics

reg_acc = metrics.accuracy_score(y_obs, y_reg_predict)
print(f'{reg_acc*100:.2f}%')

reg_precision = metrics.precision_score(y_obs, y_reg_predict)
print(f'{reg_precision*100:.2f}%')

reg_recall = metrics.recall_score(y_obs, y_reg_predict)
print(f'{reg_recall*100:.2f}%')
# %%
reg_mconf = metrics.confusion_matrix(y_obs, y_reg_predict)
reg_mconf = pd.DataFrame(reg_mconf,
                         index=['false', 'true'],
                         columns=['false', 'true'])
reg_mconf

# %%
from sklearn import tree
arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(X, y_obs)
y_arvore_predict = arvore.predict(X)

# %%
arvore_acc = metrics.accuracy_score(y_obs, y_arvore_predict)
print(f'{arvore_acc*100:.2f}%')

arvore_precision = metrics.precision_score(y_obs, y_arvore_predict)
print(f'{arvore_precision*100:.2f}%')

arvore_recall = metrics.recall_score(y_obs, y_arvore_predict)
print(f'{arvore_recall*100:.2f}%')

# %%
arvore_mconf = metrics.confusion_matrix(y_obs, y_arvore_predict)
arvore_mconf = pd.DataFrame(
  arvore_mconf,
  index=['false', 'true'],
  columns=['false', 'true']
)
arvore_mconf

# %%
from sklearn import naive_bayes
nb = naive_bayes.GaussianNB()
nb.fit(X, y_obs)
y_nb_predict = nb.predict(X)

# %%
nb_proba = nb.predict_proba(X)[:,1]
y_nb_predict = nb_proba > 0.2

nb_acc = metrics.accuracy_score(y_obs, y_nb_predict)
print(f'{nb_acc*100:.2f}%')

nb_precision = metrics.precision_score(y_obs, y_nb_predict)
print(f'{nb_precision*100:.2f}%')

nb_recall = metrics.recall_score(y_obs, y_nb_predict)
print(f'{nb_recall*100:.2f}%')

# %%
nb_mconf = metrics.confusion_matrix(y_obs, y_arvore_predict)
nb_mconf = pd.DataFrame(
  nb_mconf,
  index=['false', 'true'],
  columns=['false', 'true']
)
nb_mconf

# %%
df['proba_nb'] = nb_proba
df

# %%
import matplotlib.pyplot as plt
roc_curve = metrics.roc_curve(y_obs, nb_proba)
plt.plot(roc_curve[0], roc_curve[1])
plt.grid(True)
plt.plot([0,1], [0,1], '--')
plt.show()

# %%
roc_auc = metrics.roc_auc_score(y_obs, nb_proba)
print(f'{roc_auc*100:.2f}%')

# %%
