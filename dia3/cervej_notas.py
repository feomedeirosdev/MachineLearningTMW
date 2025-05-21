# %%
import pandas as pd
df = pd.read_excel('../datas/dados_cerveja_nota.xlsx')
df

# %%
import matplotlib.pyplot as plt
plt.plot(
  df['cerveja'], 
  df['nota'],
  linestyle='--',
  color='blue',
  marker='o',
  markerfacecolor='#1f77b4', # Cor de preenchimento (mesma azul)
  markeredgecolor='black', # Borda preta
  # label='Exemplo
  ),
plt.title("Cerveja X Notas")
plt.xlabel("quantidade de cerveja")
plt.ylabel("nota")
# plt.xlim(0, 5)
# plt.ylim(0, 20)
# plt.xticks(range(0, 6, 1))
# plt.yticks(range(0, 21, 5))
plt.grid(True)
# plt.legend(loc="upper left")
plt.show()

# %%
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df[['cerveja']], df['nota'])

# %%
import numpy as np

A = reg.intercept_
B = reg.coef_
x = np.linspace(0, 10, 100) 
y = A + B * x

# %%
plt.plot(
  df['cerveja'], 
  df['nota'],
  linestyle='',
  color='blue',
  marker='o',
  markerfacecolor='#1f77b4', # Cor de preenchimento (mesma azul)
  markeredgecolor='black', # Borda preta
  ),

plt.plot(
  x,
  y,
  color='red'
)

plt.title("Cerveja X Notas")
plt.xlabel("quantidade de cerveja")
plt.ylabel("nota")
plt.grid(True)

plt.show()

# %%
X = df[['cerveja']].drop_duplicates()
y_reg = reg.predict(X)
y_reg

plt.plot(
  df['cerveja'], 
  df['nota'],
  linestyle='',
  color='blue',
  marker='o',
  markerfacecolor='#1f77b4', # Cor de preenchimento (mesma azul)
  markeredgecolor='black', # Borda preta
  ),

plt.plot(
  x,
  y,
  color='red'
)

plt.plot(
  X,
  y_reg,
  color='green',
  linestyle=':'
)

plt.title("Cerveja X Notas")
plt.xlabel("quantidade de cerveja")
plt.ylabel("nota")
plt.grid(True)

plt.show()

# %%
from sklearn import tree
arvore = tree.DecisionTreeRegressor(max_depth=2)
arvore.fit(df[['cerveja']], df['nota'])

y_arvore = arvore.predict(X)
# %%
plt.plot(
  df['cerveja'], df['nota'],
  linestyle='', color='blue', marker='o',
  markerfacecolor='#1f77b4', markeredgecolor='black',
)
plt.plot(x, y, color='red')
plt.plot(X, y_reg, color='green', linestyle=':')
plt.plot(X, y_arvore, color='orange', linestyle='-')

plt.title("Cerveja X Notas")
plt.xlabel("quantidade de cerveja")
plt.ylabel("nota")
plt.grid(True)
plt.legend(['observado', 'y=A+B*x', 'regressão linear', 'árvore de decisão'])

plt.show()

# %%
