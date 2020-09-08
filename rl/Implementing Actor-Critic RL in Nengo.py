#!/usr/bin/env python
# coding: utf-8

# In[1]:

import nengo
import numpy as np
import matplotlib.pyplot as plt
from dv import LegacyAedatFile


# Here is an implementation of a simple actor-critic Reinforcement Learning algorithm in Nengo.
# 
# This document is based on ideas I sketched out when teaching learning rules in Nengo 5 years ago (https://github.com/tcstewar/syde556-1/blob/master/SYDE%20556%20Lecture%2010%20Learning.ipynb) and I'd implemented all of these pieces before, but this is the first time I've combined them all together into a full actor-critic RL system.  This means that I haven't done any parameter tuning and there's probably lots of room for improvement.  
# 
# In addition to the link above, I'm also indebted to these documents for (hopefully) keeping me on the right track for these algorithms.  Please let me know where I've got things wrong and correct me!
#  - https://towardsdatascience.com/introduction-to-actor-critic-7642bdb2b3d2
#  - https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
#  - http://incompleteideas.net/book/first/ebook/node66.html
#  
# The basic idea behind an actor-critic algorithm is that you are making two different predictions.  One thing you are predicting is the future value of the current state (i.e. the expected rewards you will be getting in the future, given that you are in the current state).  This is the Critic.  The other thing you are predicting is how good the different available actions are given the current state.  This is the Actor.
# 
# ## Implementing the Critic
# 
# The Critic part is pretty much the standard reinforcement learning idea.  The idea is that we have a neural network outputing this prediction, and we can update that prediction based on the reward we expect and the reward we get.  The standard trick is:
# 
# - $V(s_t) = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...$
# - $V(s_t) = r_t + \gamma V(s_{t+1})$
# 
# So, if we have a network outputting $V(s_t)$, and then we see what state we get to and what reward we actually get, then we can use that to update the neural network.
# 
# - $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
# 
# (Note: there's lots of variants of this, most noteably ones that use $Q(s,a)$ instead of $V(s)$, which learns the expected reward of doing action $a$ in state $s$)
# 
# Of course, that equation about future events, so let's shift the time by one time step so that it's about current and past events.
# 
# - $\delta_{t-1} = r_{t-1} + \gamma V(s_{t}) - V(s_{t-1})$
# 
# To implement this in Nengo, all we need is some neurons that store the state, and we compute the error signal as above and apply it to the learning rule.  For this, we're just using the PES learning rule (which in this case is just standard delta rule, so we are only updating the readout/decoder weights and no backprop is needed).  
# 
# Since the computation involves shifting things in time, we also need to figure out how to implement that delay, and to decide how long it is in actual time (should it be one time-step? But in a continuous neuron model like LIF, what does a time step really mean?).  One way to do this would be to add a delay on the `nengo.Connection` (we do this via the `synapse` parameter).  Another option is to use a low-pass filter as the synapse, and using a longer time-constant to give something like a longer delay.  This also should work well with spiking neurons, as it spreads the spike out over time a bit.  (Note: it's interesting to compare this low-pass filter to the idea of eligibility traces...).

# Now we need the learning system.  First, we need to compute the error signal, which is $r_{t-1} + \gamma V(s_{t}) - V(s_{t-1})$.  But, to do the time delays, we are using the synapses, so it's more like $h(t)\circledast r(t) + \gamma V(t) - h(t)\circledast V(t)$ (where $h(t)$ is the synapse model). This is just adding a bunch of things together that we already have, and sometimes filtering them, so here's how we compute that just by using `nengo.Connection` objects.:

# Now we need to create the connection this error signal will be applied to.  This is just the standard PES rule, which is usually pretty much just delta rule.  However, in this case we have to be a little more careful because this is the error signal for the *past* ($\delta_{t-1}$).  Normal delta rule would just be $\Delta w_i = -\alpha \delta a_i$ (where $\alpha$ is the learning rate, $\delta$ is the error signal, and $a_i$ is the output of the $i$th neuron).  But, PES also has a synaptic filter that can be applied to the neuron activity: $\Delta w_i = -\alpha \delta (h(t) \circledast a_i)$.  So, in the ideal case, if we had a delay as $h(t)$, then that would be exactly the standard RL learning rule.  We can do a delay here, or we can do the same trick of doing a low-pass filter instead to get something like a delay that might work well with spiking neurons.  We specify this for the PES rule with the `pre_synapse` parameter.

# There's lots that can be explored about this.  In addition to the number of neurons, the learning rate, and the `tau_slow` and `tau_fast` parameters, there is also things like setting `neuron_type=nengo.RectifiedLinear()` or `neuron_type=nengo.Sigmoid()` when creating the `nengo.Ensemble` would use non-spiking neurons.  You can also try an actual delay rather than a low-pass filter by making use of the following `Delay` object and setting `tau_slow = Delay(0.01)`.  One other interesting thing to try is adjusting the sparsity in the neural network by setting `intercepts=nengo.dists.Uniform(0.6,1)` when creating the `nengo.Ensemble`.  

# In[2]:


# an implementation of a pure Delay synapse for nengo
class Delay(nengo.synapses.Synapse):
    def __init__(self, delay, size_in=1):
        self.delay = delay
        super().__init__(default_size_in=size_in, default_size_out=size_in)

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        steps = int(self.delay/dt)
        if steps == 0:
            def step_delay(t, x):
                return x
            return step_delay
        assert steps > 0

        state = np.zeros((steps, shape_in[0]))
        state_index = np.array([0])

        def step_delay(t, x, state=state, state_index=state_index):
            result = state[state_index]
            state[state_index] = x
            state_index[:] = (state_index + 1) % state.shape[0]
            return result

        return step_delay


# ## Implementing the Actor
# 
# So now the system can figure out how good a particular state is (in terms of what sorts of future rewards are expected).  So how can we figure out what the right action to take would be that would lead to good states?
# 
# The fun trick in Actor-Critic learning is that it turns out you can use *the same error signal* as you're computing for the Critic part to tell you what action to take!  The basic intuition is that if you do an action and it turns out better than you expected (i.e. $\delta_{t-1} = r_{t-1} + \gamma V(s_{t}) - V(s_{t-1})$ is positive), then this means whatever action you just performed is better than you thought it was, and so you should increase your chance of doing that action in the state you were in ($s_{t-1}$).
# 
# So, all we need is to set up another `Connection` to readout how good our different actions are (one numerical value per action).  We then pass those values through a softmax to turn them into probabilities, and sample from that probability distribution to get our actual choice.  (Note: we could do all these steps using neurons in nengo, but for simplicity let's just do those steps as normal code).
# 
# Now when we compute the error for that `Connection`, we do
# 
# $\delta_{actor, i} = \delta$ if $i$ is the chosen action, otherwise $0$ 
# 
# (i.e. if you chose action $i$ and it was better than you thought it was ($\delta > 0$) then increase your chance of doing that action in the state $\delta_{actor, i}>0$)
# 
# That's the simplest form of the idea, but there's a bunch of variations.  In particular, if choose action $i$ and it's better than I expect, then not only should I increase how good I think that action is, but I should also decrease how good I think the other actions are (and vice-versa).  Furthermore, I might want to scale this based on the probability of choosing those actions.  To make this a little less ad-hoc, one common thing to do is to say something like "the loss function is $-ln(p_{chosen})\delta$.  If you go and take the derivative of that loss function with respect to the actor output weights (i.e. taking the derivative through that softmax that's happening0, you see that this is the same as the following error rule:
# 
# $\delta_{actor, i} = \delta(1-p_i)$ if $i$ is the chosen action, otherwise $-\delta p_i$ 
# 
# (Note: This is known as the advantage actor-critic algorithm)
# 
# Let's add this to the nengo model.  We'll have to slightly modify the `Environment` so that it actually accepts an action choice.  We'll have 2 actions:
# 
# - Recognize gesture 1
# - Recognize gesture 2
# 
# Recognizing a gesture either send back a positive reward if the choice is correct, or a negative reward if the choice is wrong.

# In[3]:


def load_gesture(user, gesture, packet_size):
    with open("/home/thomas/apps/grill-eprop-lsnn/DVS Gesture dataset/DvsGesture/user0"+str(user)+"_fluorescent_labels.csv", "r") as l:
        for line in l:
            labels = line.split(",")
            if labels[0] == str(gesture):
                start = int(labels[1])
                end = int(labels[2])
    
    events = []
    with LegacyAedatFile("/home/thomas/apps/grill-eprop-lsnn/DVS Gesture dataset/DvsGesture/user0"+str(user)+"_fluorescent.aedat") as f: 
        i = 0
        cumul = 0
        eslice = np.zeros((128, 128))
        for event in f:
            if event.timestamp >= start and event.timestamp <= end:
                if i == 0:
                    timeref = event.timestamp
                    i = 1
                cumul += event.timestamp - timeref
                eslice[event.x, event.y] = 1
                if cumul > packet_size:
                    cumul = 0
                    timeref = event.timestamp
                    events.append(eslice)
                    eslice = np.zeros((128, 128))
    return events

class Environment:
    def __init__(self, packet_size):
        #self.user = np.random.randint(1, 10) # We chose a random user between 1 and 9
        self.user = 1
        self.all_events = [load_gesture(self.user, 1, packet_size), load_gesture(self.user, 2, packet_size)] # We preload gesture 1 and 2 to accelerate training
        self.swap_gesture()
        self.cursor = 0
        
    def swap_gesture(self): # We chose a random gesture
        self.cursor = 0
        self.gesture = np.random.randint(2)
        self.events = self.all_events[self.gesture]
    
    def update(self, x): # update with 2 actions, the gesture chaging only when all the events have been read.
        state = self.events[self.cursor]
        self.cursor += 1
        reward = 0
        if self.cursor >= 5000:
            print("gesture complete", self.gesture, x)
            if x[0] > 0: # We decide this is the 1st gesture
                reward = 1 if self.gesture == 0 else -1
            elif x[1] > 0: # We decide this is the 2nd gesture
                reward = 1 if self.gesture == 1 else -1
            self.swap_gesture()
                
        return *state.flatten(), reward
    
#    def update(self, x): # update with 3 actions, the gesture changing when using action 2 and 3.
#        state = self.events[self.cursor]
#        if x[0] > 0: # We choose to do nothing and wait for more inputs
#            reward = 0
#        elif x[1] > 0: # We decide this is the 1st gesture
#            self.swap_gesture()
#            reward = 1 if self.gesture == 0 else -1
#        elif x[2] > 0: # We decide this is the 2nd gesture
#            self.swap_gesture()
#            reward = 1 if self.gesture == 1 else -1
#            
#        self.cursor += 1
#        if self.cursor > len(self.events):
#            self.cursor = 0
#        return *state.flatten(), reward


# And now let's re-create the Critic part of the model as well as the new Actor part.

# In[21]:


tau_slow = 0.01
tau_fast = None
discount = 0.95

class Sizes:
    edge_dvs = 128
    edge_sptc = 32
    pop_dvs = edge_dvs ** 2
    pop_sptc = edge_sptc ** 2
    actions = 2

weights = np.zeros((1024, 16384))
k = 0
for i in range(0, 128, 4):
    for j in range(0, 128, 4):
        temp = np.zeros((128, 128))
        temp[i:i+4, j:j+4] = 1
        weights[k] = temp.flatten()
        k += 1
    
packet_size = 10000

environment = Environment(packet_size)
model = nengo.Network()
with model:
    
    # create the environment
    #   it has 16385 inputs: the first 16384 are the state made of the 128*128 pixels
    #   array mapped from the DVS sensor, and the last is the reward.
    #   there is 2 action possible, recognize gesture 1 and recognize gesture 2.
    env = nengo.Node(lambda t, x: environment.update(x), size_in=Sizes.actions, size_out=Sizes.pop_dvs+1)
    
    # set up some other Nodes that just grab the state and reward information,
    #  just for clarity
    state = nengo.Node(None, size_in=Sizes.pop_dvs)
    nengo.Connection(env[:-1], state, synapse=None)
    reward = nengo.Node(None, size_in=1)
    nengo.Connection(env[-1], reward, synapse=None)
    
    # create the neural network to encode the state. The default is LIF neurons.
    input_reduction = nengo.Ensemble(n_neurons=Sizes.pop_sptc, dimensions=Sizes.pop_sptc)
    nengo.Connection(state, input_reduction, synapse=None, transform=weights)
    
    # layer of LIF neurons to learn gesture differentiation.
    learning_ens = nengo.Ensemble(n_neurons=128, dimensions=input_reduction.n_neurons)
    nengo.Connection(input_reduction, learning_ens, synapse=None)
    
    # this is the output value that the critic will learn
    value = nengo.Node(None, size_in=1)

    # record the value and the reward
    p_value = nengo.Probe(value)
    p_reward = nengo.Probe(reward)
    
    # compute the critic error
    value_error = nengo.Node(None, size_in=1)
    nengo.Connection(value, value_error, transform=-discount, synapse=tau_fast)
    nengo.Connection(value, value_error, synapse=tau_slow)
    nengo.Connection(reward, value_error, transform=-1, synapse=tau_slow)
    
    # make the connection to learn on
    c = nengo.Connection(learning_ens.neurons, value, transform=np.zeros((1, learning_ens.n_neurons)), 
                         learning_rule_type=nengo.PES(learning_rate=1e-4, pre_synapse=tau_slow))
    # connect the error signal to the learning rule
    nengo.Connection(value_error, c.learning_rule, synapse=None)

    ### HERE IS THE NEW STUFF FOR THE ACTOR
    
    # the raw output from the network
    raw_actions = nengo.Node(None, size_in=Sizes.actions, label='raw_actions')
    
    # compute the softmax
    def softmax(t, x):
        return np.exp(x)/np.sum(np.exp(x))
    actions = nengo.Node(softmax, size_in=Sizes.actions, label='actions')
    nengo.Connection(raw_actions, actions, synapse=None)

    # do the random sampling and output which action we're taking
    #  (here I just decided to represent choosing the action as +1 and not choosing as -1)
    def choice_func(t, x):
        c = np.random.choice(np.arange(Sizes.actions), p=x)
        result = np.full(Sizes.actions, -1)
        result[c] = 1
        return result
    choice = nengo.Node(choice_func, size_in=Sizes.actions, size_out=Sizes.actions, label='choice')
    nengo.Connection(actions, choice, synapse=None)
    
    # and now connect the choice to the environment
    nengo.Connection(choice, env, synapse=None)
    
    # and here is the computation of the error signal
    c_actor = nengo.Connection(learning_ens.neurons, raw_actions, transform=np.zeros((Sizes.actions, learning_ens.n_neurons)), 
                               learning_rule_type=nengo.PES(learning_rate=1e-4, pre_synapse=tau_slow))

    # implement the advantage actor-critic error rule
    #  the Node gets 2*nb_action+1 inputs: the delta for the critic part, "nb_action" values indicating which action was chosen
    #  (+1 for the chosen and -1 for the non-chosen ones), and the choice probabilities for the nb_action actions
    def actor_error_func(t, x):
        delta = x[0]
        chosen = x[1:Sizes.actions+1]
        prob = x[Sizes.actions+1:2*Sizes.actions+1]
        # compute the error
        e = np.where(chosen>0, delta*(1-prob), -delta*prob)
        return e

    actor_error = nengo.Node(actor_error_func, size_in=2*Sizes.actions+1, label='actor_error')
    nengo.Connection(value_error, actor_error[0], synapse=None)
    nengo.Connection(choice, actor_error[1:Sizes.actions+1], synapse=None)
    nengo.Connection(actions, actor_error[Sizes.actions+1:2*Sizes.actions+1], synapse=None)
    nengo.Connection(actor_error, c_actor.learning_rule, transform=-1, synapse=tau_slow) 
    
    p_choice = nengo.Probe(choice)   # record the actual choices
    p_prob = nengo.Probe(actions)    # record the probabilities 


# Now let's run the model and see what happens.

# In[22]:


sim = nengo.Simulator(model)


# In[ ]:


sim.run(100)


# In[7]:


plt.figure(figsize=(14,5))
plt.plot(sim.trange(), sim.data[p_value], label='value')
plt.plot(sim.trange(), sim.data[p_reward]/1000, label='reward')
plt.legend()
plt.show()


# In[12]:


np.save("val_learning2", sim.data[p_value])


# ## Things to explore
# 
# While this shows the general approach works, there are many things to explore to see how they affect the model.
# 
# First, there's all the parameters that we mentioned when just looking at the Critic part:
#  - number of neurons
#  - neuron type (spiking, ReLU, sigmoid, etc.)
#  - `tau_fast`, `tau_slow`, `discount`
#  - `learning_rate` (for both actor and critic)
#  - various versions of actor-critic (i.e. the error computation for the actor learning)
#  
# Second, there's the application to different tasks.  The two next things to try would be the standard cart-pole and mountain car RL test cases.  However, this is a bit tricky because you have to decide how to reset the run and make sure that it's not learning between runs.  I prefer tasks that are always continuing, like this one.  Indeed, I might even be tempted to jump straight up to the Atari Pong example, since that can be more about a very long continuous run.
# 
# For these more complex tasks, we would want to add some recurrence to the network.  The easiest way to do that would be to add a recurrent connection on the `nengo.Ensemble` that implements the LMU.  That way it's already a good recurrent network for storing information over time, and we don't have to do any learning in the recurrence.  We could also do a standard reservoir by having a random recurrent connection.  It would also be interesting to try implementing e-prop or another learning algorithm, if we can get those working.
# 
# I mentioned the representation sparsity earlier, and I also think that will be an important parameter to work with, especially given that this sort of online learning exhibits a lot less catestrophic forgetting given a sparser neural representation.  There are a variety of ways of imposing sparsity in Nengo, but the easiest is what I mentioned above with adjusting the intercepts.  This seems to affect the Critic part of the algorithm a lot, but I haven't explored it in the Actor.
