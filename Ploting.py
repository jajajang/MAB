import ChangeDetection as cd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import csv
import pandas as pd







sns.lineplot(x="time", y="subopt", data=resulty, ci=95, label='UCB1_reuse')
sns.lineplot(x="time", y="subopt", data=comp_resulty, ci=95, label='UCB1_reset')

plt.title('Regret analysis : Reuse vs Reset')
plt.legend()
plt.xlabel('Time')
plt.ylabel('expected regret')
#plt.axvline(x=T/10, color='r', label = 'dist. change')
plt.show()