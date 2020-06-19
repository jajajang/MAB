import argparse
import ChangeDetection as cd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import csv
import pandas as pd


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--alpha', nargs='+', type=float, default=0.8, metavar='N',
                    help='alpha_value')
parser.add_argument('--ptop', type=float, default=0.8, metavar='N',
                    help='bernoulli probability top')
parser.add_argument('--pbot', type=float, default=0.2, metavar='N',
                    help='bernoulli probability bottom')
parser.add_argument('--repeat', type=int, default=5, metavar='N',
                    help='Number of repeated experiments')
parser.add_argument('--delta', nargs='+', type=float, default=0.1, metavar='N',
                    help='large delta - 기저에서 빠질 값')
parser.add_argument('--factor', nargs='+', type=float, default=0.5, metavar='N',
                    help='gamma에 곱해질 factor')
parser.add_argument('--timescale', nargs='+', type=int, default=100000, metavar='N',
                    help='총 시간')

args=parser.parse_args()


def triangle_wave(t,delta, low, high):
    peri=2*(high-low)/delta
    return (high-low)*2*np.abs(t/peri-np.floor(t/peri+0.5))+low


k = 2
p_top =args.ptop
p_bot =args.pbot
Times=args.timescale
alphy = args.alpha
rep = args.repeat
factory = args.factor
total_exp=len(Times)*rep
delty = args.delta  # large Delta..... 빼서 계산할 예정
def dis_time(t,gamma):
    return (1-gamma**t)/(1-gamma)

for alpha in alphy:
    for D in delty:
        for factor in factory:
            record_final_regret = [0] * total_exp
            record_total_regret = [0] * total_exp
            for z in range(len(Times)):
                T=Times[z]
                delta = T ** (-1 * alpha)
                gamma = 1 - 1 / (T**(alpha*factor))
                Regret_hist = 0
                Regret_hist_list = np.zeros(T)
                Total_regret = 0
                Total_regret_list = np.zeros(T)

                check_parser = f'Order_alpha_{alpha}_ptop_{p_top}_pbot_{p_bot}_delta_{D}_factor_{factor}'
                print(check_parser)

                #w = 4*np.floor(T ** (alpha / 2)).astype(np.int)
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
                    dis_actval=np.zeros((2,1))
                    dis_reward=np.zeros((2,1))
                    dis_ucb = np.zeros(k)

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
                                ucb[i] = dis_reward[i]/dis_actval[i] + np.sqrt(0.5*np.log(dis_time(t, gamma))/dis_actval[i])
            #                    ucb[i] = np.sum(reward_list[i, -w:]) / np.sum(action_list[i, -w:]) + np.sqrt(
            #                        2 * np.log(t) / np.sum(action_list[i, -w:]))
                                action = np.argmax(ucb)

                        current_action = np.zeros((2,1))
                        current_reward = np.zeros((2,1))
                        current_action[action] += 1

                        virtual_reward[0] = np.random.binomial(n=1, p=p[0])
                        virtual_reward[1] = np.random.binomial(n=1, p=p[1])
                        current_reward[action] += virtual_reward[action]

                        action_list = np.c_[action_list, current_action]
                        reward_list = np.c_[reward_list, current_reward]

                        dis_actval = dis_actval * gamma + current_action
                        dis_reward = dis_reward * gamma + current_reward

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
                    record_final_regret[z*rep+s]=np.sum(minus_Delta_plot)
                    record_total_regret[z*rep+s]=Exp_Regret


                timeliness = ['time']+wholetime
                plotspacess = ['subopt']+plotspace
                eachregretss = ['each_regret']+each_regret_recorder
                minusdeltass = ['minus_Delta']+minus_Delta_recorder
                finrec=f'final_record_alpha_{alpha}_delta_{D}_factor{factor}'
                finalrecord = [finrec]+record_final_regret
                totalrecord = ['total_record']+record_total_regret

                #rows = zip(timeliness, plotspacess, eachregretss, minusdeltass)
                rows = zip(finalrecord, totalrecord)
                file1name=f'Order_T_{T}_alpha_{alpha}_ptop_{p_top}_pbot_{p_bot}_delta_{D}_factor_{factor}_discount_result_only.csv'
                with open(file1name, "w", newline='') as f:
                    writer = csv.writer(f)
                    for row in rows:
                        writer.writerow(row)
                    print('minusDelta : ')
                    print(finalrecord)
                    print('totalRecord : ')
                    print(totalrecord)

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
