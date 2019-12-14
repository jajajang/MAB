import numpy as np

def changeDetection(R, w, bound_const):
    if len(R) < w:
        return False;
    R0 = R[len(R)-w:]
    early = R0[:w // 2]
    later = R0[w // 2:]
    if np.abs(np.sum(early)-np.sum(later))>bound_const:
        return True;
    return False;

def filterList(reward, action):
    bool_list = list(map(bool, action))
    return reward[bool_list]

"""
stopit=False
reward_trail=[]
time=0
pee=0.5
wind=800

while not stopit:
    if time>wind:
        pee=0.6
    reward = np.random.binomial(n=1, p=pee)
    reward_trail.append(reward)
    if len(reward_trail)>wind:
        reward_trail.pop(0)
    stopit=changeDetection(reward_trail,w=wind, bound_const=50)
    time+=1

print(time)
"""