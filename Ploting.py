import ChangeDetection as cd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import csv
import pandas as pd





resulty = pd.read_csv('C:/Users/KyoungseokJang/Documents/GitHub/MAB/MutualInfo_binMNIST.csv')


sns.lineplot("subopt", data=resulty)

plt.title('Regret analysis : Reuse vs Reset')
plt.legend()
plt.xlabel('Time')
plt.ylabel('expected regret')
#plt.axvline(x=T/10, color='r', label = 'dist. change')
plt.show()