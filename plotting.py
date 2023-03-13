#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns
import pandas as pd
# %%
train_acc = loadmat("train.mat")['tr_acc']
e = np.arange(train_acc.shape[1])
plt.plot(e, train_acc.ravel())
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Small Test Dataset 3.5")
# %%
train_acc = loadmat("train_b72.mat")['tr_acc']
val_acc = loadmat("train_b72.mat")['val_acc']
e = np.arange(train_acc.shape[1])
plt.plot(e, train_acc.ravel())
plt.plot(e, val_acc.ravel())
plt.legend(['Train', 'Dev'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("All Test Dataset 3.6")
# %%
dev_1e4 = loadmat("train_b72.mat")['val_acc'][:, :10]
dev_5e4 = loadmat("train_b72_lr5e4.mat")['val_acc']
dev_1e3 = loadmat("train_b72_lr1e3.mat")['val_acc']
e = np.arange(dev_1e4.shape[1])
plt.plot(e, dev_1e4.ravel())
plt.plot(e, dev_5e4.ravel())
plt.plot(e, dev_1e3.ravel())
plt.legend(['1e-4', '5e-4', '1e-3'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Parameter Sweep 3.7")
# %%
dbu_vacc = loadmat("train_b72.mat")['val_acc'][:, :10]
dbu_tacc = loadmat("train_b72.mat")['test_acc'][:, :10]
bbu_vacc = loadmat("train_b32_bbu_lr1e4.mat")['val_acc']
bbu_tacc = loadmat("train_b32_bbu_lr1e4.mat")['test_acc']
bbc_vacc = loadmat("train_b32_bbc_lr1e4.mat")['val_acc']
bbc_tacc = loadmat("train_b32_bbc_lr1e4.mat")['test_acc']
# %%
e = np.arange(dbu_tacc.shape[1])
plt.plot(e, dbu_vacc.ravel())
plt.plot(e, bbu_vacc.ravel())
plt.plot(e, bbc_vacc.ravel())
plt.legend(['distilBERT-base-uncased', 'BERT-base-uncased', 'BERT-base-cased'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Parameter Sweep 3.8")
# %%
data = pd.DataFrame({'models': ['distilBERT-base-uncased', 'BERT-base-uncased', 'BERT-base-cased', 'distilBERT-base-uncased', 'BERT-base-uncased', 'BERT-base-cased'],
                     'Accuracy':[dbu_vacc[0, 4], bbu_vacc[0, 6], bbc_vacc[0, 4], dbu_tacc[0, 4], bbu_tacc[0, 6], bbc_tacc[0, 4]], 
                     'type':['Validation', 'Validation', 'Validation', 'Testing', 'Testing', 'Testing']})
sns.barplot(data=data, x="models", y="Accuracy", hue="type")
# %%
