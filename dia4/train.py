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
  X, # Variáveis preditivas ou caracteristicas
  y, # Variaável alvo ou objetivo
  test_size = 0.2,
  random_state = 42,
  stratify = y
))

# %%
print(f'Tx resposta y treino: {y_train.mean()*100:.2f}%')
print(f'Tx resposta y test: {y_test.mean()*100:.2f}%')

# %%
X_train.isna().sum()

# %%
# Tratando os dados nulos
imput_avgRecorrencia = X_train['avgRecorrencia'].max()
X_train['avgRecorrencia'] = X_train['avgRecorrencia'].fillna(imput_avgRecorrencia)
X_test['avgRecorrencia'] = X_test['avgRecorrencia'].fillna(imput_avgRecorrencia)

# %%
from sklearn import tree
from sklearn import metrics

decision_tree_model = tree.DecisionTreeClassifier(
  max_depth=5,
  min_samples_leaf=50,
  random_state=42
)
decision_tree_model.fit(X_train, y_train)

# %%
# Predição
y_tree_pred_train = decision_tree_model.predict(X_train)

print('Árvore de Decisão (TREINO)')
# Acurácia
tree_acc_train = metrics.accuracy_score(y_train, y_tree_pred_train)
print(f'Acurácia: {tree_acc_train*100:.2f}%')
# Precisão
tree_precision_train = metrics.precision_score(y_train, y_tree_pred_train)
print(f'Precisão: {tree_precision_train*100:.2f}%')
# Recall
tree_recall_train = metrics.recall_score(y_train, y_tree_pred_train)
print(f'Recall: {tree_recall_train*100:.2f}%')

# %%
tree_mconf_train = metrics.confusion_matrix(y_train, y_tree_pred_train)
df_tree_mconf_train = pd.DataFrame(
  tree_mconf_train,
  index=['false', 'true'],
  columns=['false', 'true']
)
df_tree_mconf_train

# %%
import matplotlib.pyplot as plt

y_tree_proba_train = decision_tree_model.predict_proba(X_train)[:,1]
roc_curve = metrics.roc_curve(y_train, y_tree_proba_train)

roc_auc_train = metrics.roc_auc_score(y_train, y_tree_proba_train)
print(f'ROC_AUC: {roc_auc_train*100:.2f}%')

plt.plot(roc_curve[0], roc_curve[1], linestyle='-', marker='o')
plt.grid(True)
plt.show()

# %%
y_tree_predict_test = decision_tree_model.predict(X_test)
tree_acc_test = metrics.accuracy_score(y_test, y_tree_predict_test)
tree_precision_test = metrics.precision_score(y_test, y_tree_predict_test)
tree_recall_test = metrics.recall_score(y_test, y_tree_predict_test)

print('Árvore de Decisão (TESTE)')
print(f'Acurácia: {tree_acc_test*100:.2f}%')
print(f'Precisão: {tree_precision_test*100:.2f}%')
print(f'Recall: {tree_recall_test*100:.2f}%')

# %%
tree_mconf_test = metrics.confusion_matrix(y_test, y_tree_predict_test)
df_tree_mconf_test = pd.DataFrame(
  tree_mconf_test,
  index=['false', 'true'],
  columns=['false', 'true']
)

# %%
y_tree_proba_test = decision_tree_model.predict_proba(X_test)[:,1]
roc_curve_test = metrics.roc_curve(y_test, y_tree_proba_test)

roc_auc_test = metrics.roc_auc_score(y_test, y_tree_proba_test)
print(f'ROC_AUC: {roc_auc_test*100:.2f}%')

plt.plot(roc_curve_test[0], roc_curve_test[1], linestyle='-', marker='o')
plt.show()

