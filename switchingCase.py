import ChangeDetection as cd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import csv
import pandas as pd

k = 2
p0 = [0.2, 0.5]
p1 = [0.8, 0.5]
T=100000
changeT=5000
rep=30
Exp_Regret = 0
Exp_Regret_plot = np.zeros(T)
comp_Exp_Regret = 0
comp_Exp_Regret_plot = np.zeros(T)
arm_choice = np.zeros((2,T))
reset_l=50
w=800
b=50
plotspace=[]
wholetime=[]
comp_plotspace=[]
delayspace=[]
comp_delayspace=[]

for s in range(0,rep):

    Exp_Regret = 0
    Exp_Regret_plot = np.zeros(T)
    comp_Exp_Regret = 0
    comp_Exp_Regret_plot = np.zeros(T)
    delay_list=[]
    comp_delay_list=[]
    '''
    UCB1 algorithm
    '''
    action_list=np.zeros((2,1))
    reward_list=np.zeros((2,1))
    current_action=np.zeros((2,1))
    current_reward=np.zeros((2,1))
    ucb = np.zeros(k)

    comp_action_list=np.zeros((2,1))
    comp_reward_list=np.zeros((2,1))
    comp_current_action=np.zeros((2,1))
    comp_current_reward=np.zeros((2,1))
    comp_ucb = np.zeros(k)

    virtual_reward=np.zeros((2,1))

    p=p0
    t0=0
    c_t0=0
    break_t=0
    for t in range(0,T):
        if np.sum(action_list[0, :])==0:
            action=0
        elif np.sum(action_list[1, :])==0:
            action=1
        else:
            for i in range(0, k):
                ucb[i] = np.sum(reward_list[i, :]) / np.sum(action_list[i, :]) + np.sqrt(
                    2 * np.log(t-t0) / np.sum(action_list[i, :]))
                action = np.argmax(ucb)

        if np.sum(comp_action_list[0, :])==0:
            comp_action=0
        elif np.sum(comp_action_list[1, :])==0:
            comp_action=1
        else:
            for i in range(0, k):
                comp_ucb[i] = np.sum(comp_reward_list[i, :])/np.sum(comp_action_list[i, :]) + np.sqrt(2*np.log(t-c_t0)/np.sum(comp_action_list[i, :]))
            comp_action = np.argmax(comp_ucb)

        current_action = np.zeros((2,1))
        current_reward = np.zeros((2,1))
        current_action[action] += 1

        comp_current_action = np.zeros((2,1))
        comp_current_reward = np.zeros((2,1))
        comp_current_action[comp_action]+=1

        ##  Breakpoint change
        if t % changeT == 0:
            break_t=t
            if p[0] == p0[0]:
                p = p1
            else:
                p = p0

        virtual_reward[0] = np.random.binomial(n=1, p=p[0])
        virtual_reward[1] = np.random.binomial(n=1, p=p[1])
        current_reward[action] += virtual_reward[action]
        comp_current_reward[comp_action] += virtual_reward[comp_action]

        action_list = np.c_[action_list, current_action]
        reward_list = np.c_[reward_list, current_reward]

        comp_action_list = np.c_[comp_action_list, comp_current_action]
        comp_reward_list = np.c_[comp_reward_list, comp_current_reward]

        Exp_Regret += np.max(p) - p[action]
        Exp_Regret_plot[t] = Exp_Regret

        comp_Exp_Regret += np.max(p) - p[comp_action]
        comp_Exp_Regret_plot[t] = comp_Exp_Regret

        if cd.changeDetection(cd.filterList(reward_list[0,:],action_list[0,:]),w=w,bound_const=b) or cd.changeDetection(cd.filterList(reward_list[1,:],action_list[1,:]),w=w,bound_const=b):
            reward_list = np.delete(reward_list, range(0, len(reward_list[0, :]) - reset_l), axis=1)
            action_list = np.delete(action_list, range(0, len(action_list[0, :]) - reset_l), axis=1)
            print('Change Detected : Time %d' %t)
            print(len(reward_list[0,:]))
            delay_list.append(t-break_t)
            t0=t+1-reset_l

        if cd.changeDetection(cd.filterList(comp_reward_list[0,:],comp_action_list[0,:]), w=w,bound_const=b) or cd.changeDetection(cd.filterList(comp_reward_list[1,:],comp_action_list[1,:]),w=w, bound_const=b):
            comp_action_list = np.zeros((2,1))
            comp_reward_list = np.zeros((2,1))
            c_t0=t+1
            comp_delay_list.append(t-break_t)
            print('comp_Change Detected : Time %d' %t)


    timeline = range(0,T)
    wholetime.extend(timeline)
    plotspace.extend(Exp_Regret_plot)
    comp_plotspace.extend(comp_Exp_Regret_plot)
    delayspace.extend(delay_list)
    comp_delayspace.extend(comp_delay_list)


timeliness = ['time']+wholetime
plotspacess = ['subopt']+plotspace
comp_plotspacess = ['subopt']+comp_plotspace

rows = zip(timeliness, plotspacess)
comp_rows = zip(timeliness, comp_plotspacess)

print(delayspace)

with open('delay'+str(reset_l)+'.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(delayspace)
with open('comp_delay'+str(reset_l)+'.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(delayspace)
with open('eggs'+str(reset_l)+'.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
with open('comp_eggs'+str(reset_l)+'.csv', "w") as f:
    writer = csv.writer(f)
    for row in comp_rows:
        writer.writerow(row)
resulty=pd.read_csv('eggs'+str(reset_l)+'.csv')
comp_resulty=pd.read_csv('comp_eggs'+str(reset_l)+'.csv')


sns.lineplot(x="time", y="subopt", data=resulty, ci=95, label='UCB1_reuse')
sns.lineplot(x="time", y="subopt", data=comp_resulty, ci=95, label='UCB1_reset')

plt.title('Regret analysis : Reuse vs Reset')
plt.legend()
plt.xlabel('Time')
plt.ylabel('expected regret')
#plt.axvline(x=T/10, color='r', label = 'dist. change')
plt.show()

