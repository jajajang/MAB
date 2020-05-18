import ChangeDetection as cd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import csv
import pandas as pd

def triangle_wave(t,delta, low, high):
    peri=2*(high-low)/delta
    return (high-low)*2*np.abs(t/peri-np.floor(t/peri+0.5))+low

k = 2
p_top =0.8
p_bot =0.2
Times=[5000, 10000, 20000, 40000, 80000, 160000]
alpha = 0.5

for z in range(0,len(Times)):
    T = Times[z]
    delta = T ** (-1 * alpha)

    Regret_hist = 0
    Regret_hist_list = np.zeros(T)
    Total_regret = 0
    Total_regret_list = np.zeros(T)

    rep = 2
    w = 2 * np.floor(T ** (alpha / 2)).astype(np.int)
    D = 0.1
    plotspace = []
    wholetime = []
    each_regret_recorder = []
    minus_Delta_recorder = []
    for s in range(0,rep):

        Exp_Regret = 0
        Exp_Regret_plot = np.zeros(T)
        each_regret_plot = np.zeros(T)
        minus_Delta_plot = np.zeros(T)
        '''
        SW-UCB1 algorithm
        '''
        action_list=np.zeros((k,1))
        reward_list=np.zeros((k,1))
        current_action=np.zeros((k,1))
        current_reward=np.zeros((k,1))
        ucb = np.zeros(k)

        virtual_reward=np.zeros((2,1))

        for t in range(0,T):
            each_regret = 0
            p=[triangle_wave(t,delta,p_bot,p_top), p_top-triangle_wave(t,delta,p_bot,p_top)]
            if np.sum(action_list[0, :])==0:
                action=0
            elif np.sum(action_list[1, :])==0:
                action=1
            else:
                for i in range(0, k):
                    ucb[i] = np.sum(reward_list[i, -w:]) / np.sum(action_list[i, -w:]) + np.sqrt(
                        2 * np.log(t) / np.sum(action_list[i, -w:]))
                    action = np.argmax(ucb)

            current_action = np.zeros((2,1))
            current_reward = np.zeros((2,1))
            current_action[action] += 1

            virtual_reward[0] = np.random.binomial(n=1, p=p[0])
            virtual_reward[1] = np.random.binomial(n=1, p=p[1])
            current_reward[action] += virtual_reward[action]

            action_list = np.c_[action_list, current_action]
            reward_list = np.c_[reward_list, current_reward]

            each_regret =np.max(p)-p[action]
            Exp_Regret += each_regret
            each_regret_plot[t]= each_regret
            Exp_Regret_plot[t] = Exp_Regret
            minus_Delta_plot[t] = np.max([each_regret-D, 0])

        timeline = range(0,T)
        wholetime.extend(timeline)
        plotspace.extend(Exp_Regret_plot)
        each_regret_recorder.extend(each_regret_plot)
        minus_Delta_recorder.extend(minus_Delta_plot)


    timeliness = ['time']+wholetime
    plotspacess = ['subopt']+plotspace
    eachregretss = ['each_regret']+each_regret_recorder
    minusdeltass = ['minus_Delta']+minus_Delta_recorder

    rows = zip(timeliness, plotspacess, eachregretss, minusdeltass)

    file1name=f'Order_T_{T}_alpha_{alpha}_ptop_{p_top}_pbot_{p_bot}.csv'
    with open(file1name, "w", newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

"""
resulty=pd.read_csv(file1name)
sns.lineplot(x="time", y="subopt", data=resulty, ci=52, label='UCB1_reuse')
sns.lineplot(x="time", y="subopt", data=dis_resulty, ci=52, label='UCB1_reset')

plt.title('Regret analysis : Reuse vs Reset')
plt.legend()
plt.xlabel('Time/10')
plt.ylabel('expected regret')
#plt.axvline(x=T/10, color='r', label = 'dist. change')
plt.show()
"""