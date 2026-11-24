"""DRPO motivation ablation, NOT the exact published DRPO optimizer.
Demonstrates that naïve group-normalization of accuracy-length rewards can assign a
negative advantage to a correct (but longer) rollout. Positive-only length
preference avoids comparing correct examples against incorrect examples.
"""
import json,math

def standard(rewards):
    mu=sum(rewards)/len(rewards); var=sum((r-mu)**2 for r in rewards)/len(rewards)
    return [(r-mu)/(math.sqrt(var)+1e-8) for r in rewards]

def positive_length_weights(correct,lengths,temp=35):
    indices=[i for i,c in enumerate(correct) if c]
    if not indices:return [0.]*len(correct)
    z=[-lengths[i]/temp for i in indices]
    exp=[math.exp(v-max(z)) for v in z]
    out=[0.]*len(correct)
    for i,v in zip(indices,exp):out[i]=v/sum(exp)
    return out

# synthetic 4 rollouts, one correct/long and one correct/short
correct=[1,1,0,0];length=[200,40,70,50]
rewards=[float(c)-0.005*l for c,l in zip(correct,length)]
print(json.dumps({"example":"synthetic only, not model evaluation","correct":correct,"length":length,"accuracy_minus_length_reward":rewards,"standard_group_advantage":standard(rewards),"positive_only_length_preference":positive_length_weights(correct,length)},indent=2))
