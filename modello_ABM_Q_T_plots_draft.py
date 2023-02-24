import numpy as np
import math as m
import statistics as stat
import matplotlib.pyplot as plt
import random as rnd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from tqdm import tqdm
import copy
import time
from statistics import pvariance, variance, stdev
import seaborn as sns

in_features = 3


class DQN(nn.Module):
    '''
    fully connected NN with 30 nodes each layer, for 6 layers
    '''
    def __init__(self, in_size, hidden_layers_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_layers_size)
        self.fc2 = nn.Linear(hidden_layers_size, hidden_layers_size)
        self.fc3 = nn.Linear(hidden_layers_size, hidden_layers_size)
        self.fc4 = nn.Linear(hidden_layers_size, hidden_layers_size)
        self.fc5 = nn.Linear(hidden_layers_size, 1)
        #self.float()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))

        return self.fc5(x)

class Ambiente():

    def __init__(self, S0 = 10, mu = 0, kappa = 5, theta = 1, sigma = 0.0001, lambd = 0.1, t0 = 0, t = 1, T = 3600, inv = 20): #T = 1.0, M = 7200, I = 1_000
        
        self.S0 = S0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.dt = 1/T
        self.T = T
        self.t0 = t0
        self.tau = t-t0
        self.lambd = lambd
        #self.numIt = numIt
        self.initial_capital = inv

    def abm(self, seed = 14, numIt=10):
        '''
        returns a matrix of Arithmetic Brownian Motion paths
        '''
        N = self.T
        I = numIt
        dt= 1.0 / self.T
        X =np.zeros((N + 1, I) ,dtype=float)
        X[0] = self.S0
        np.random.seed(seed)
        for i in range(N):
        
            X[i + 1] = X[i] + self.mu * dt + self.sigma * np.sqrt(dt) * np.random.standard_normal(I) 
    
        return np.abs(X)

    def inventory_action_transform(self, q, x):

        q_0 = self.initial_capital + 1

        q = q / q_0 - 1
        x = x / q_0
        r = m.sqrt(q ** 2 + x ** 2)
        theta = m.atan((-x / q))
        z = -x / q

        if theta <= m.pi / 4:
            r_tilde = r * m.sqrt((pow(z, 2) + 1) * (2 * (m.cos(m.pi / 4 - theta)) ** 2))

        else:

            r_tilde = r * m.sqrt((pow(z, -2) + 1) * (2 * (m.cos(theta - m.pi / 4)) ** 2))

        return 2 * (-r_tilde * m.cos(theta)) + 1, 2 * (r_tilde * m.sin(theta)) - 1

    def time_transform(self, t):

        tc = (5 - 1) / 2
        return (t - tc) / tc

    def qdr_var_normalize(self, qdr_var, min_v, max_v):

        middle_point = (max_v + min_v) / 2
        half_length = (max_v - min_v) / 2

        qdr_var = (qdr_var - middle_point) / half_length

        return qdr_var

    def price_normalise(self, price, min_p, max_p):

        middle_point = (max_p + min_p) / 2
        half_length = (max_p - min_p) / 2

        price = (price - middle_point) / half_length

        return price

    def normalise(self, inventory, time, x):
        '''
        performs the normalisation in the range [-1,1] for the feature of the NN
        '''
        q, x = self.inventory_action_transform(inventory, x)
        t = self.time_transform(time)
        #p = self.price_normalise(price, min_p, max_p)
        return q, t,  x#p,

class ReplayMemory():
    '''
    Experience replay memory
    '''

    def __init__(self, capacity):

        self.memory = deque([], maxlen=capacity)

    def add(self, inv, time, x, next_inv, next_time, reward): #inv, time, price, var, x, next_state, reward state, action, next_state, reward
        
        self.memory.append([inv, time,  x, next_inv, next_time, reward])

    def sample(self, batch_size):
        
        return rnd.sample(self.memory, batch_size)

    def __len__(self):
        
        return len(self.memory)

class Agente():

    def __init__(self, inventario, numTrain):

        self.train = numTrain
        self.memory = ReplayMemory(15_000)   
        self.env = Ambiente()
        self.main_net = DQN(in_size=in_features, hidden_layers_size=30)
        self.target_net = DQN(in_size=in_features, hidden_layers_size=30)

        for p in self.target_net.parameters():
            p.requires_grad = False

        #self.initial_capital = inventory

        self._update_target_net()

        self.learning_rate = 0.01
        self.optimizer = optim.Adam(params=self.main_net.parameters(), lr=self.learning_rate)
        self.time_subdivisions = 5
        self.inventory = inventario
        self.a_penalty = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.batch_size = 264
        self.gamma = 1#.99   ##############################################
        self.timestep = 0
        self.update_target_steps = 10000
        self.lots_size = 100
        self.matrix = np.zeros((self.inventory + 1,5))
        self.matrix_heat = np.zeros((self.inventory + 1,5))

    def _update_target_net(self):
        '''
        private method of the class: it refreshes the weight matrix of the target NN 
        '''

        self.target_net.load_state_dict(self.main_net.state_dict())

    def eval_Q(self, state, act, p_min, p_max, type = 'tensor', net = 'main'):
        '''
        Evaluates the Q-function
        '''
        if type == 'scalar' :

            q, t, x = Ambiente().normalise(state[0], state[1], act)
            in_features = torch.tensor([q, t , x], dtype=torch.float)

        if type == 'tensor':

            features = []

            for i in range(len(state)):

                q, t,  x = Ambiente().normalise(state[i][0], state[i][1], act[i])
                features.append(torch.tensor([q, t , x], dtype=torch.float))

            in_features = torch.stack(features)
            in_features.type(torch.float)

        if net == 'main':

            retval = self.main_net(in_features).type(torch.float)
            return retval

        elif net == 'target':

            retval = self.target_net(in_features).type(torch.float)
            return retval

    def q_action(self, state, min_p, max_p):
        '''
        Chooses the best action by argmax_x Q(s,x)
        '''
        features = []
        with torch.no_grad():

            for i in range(int(state[0] + 1)): # prevedi loop con stati negativi

                q, t , x = Ambiente().normalise(state[0], state[1], i) 
                features.append(torch.tensor([q, t , x], dtype=torch.float))

            qs_value = self.main_net.forward(torch.stack(features))
            action = torch.argmax(qs_value).item()

            return round(action)

    def action(self, state, min_p, max_p):
        '''
        does the exploration in the action space
        eps >= U(0,1) then tosses a coin -> 50%prob does TWAP, 50%prob does all in
        eps <= U(0,1) does the optimal Q action
        '''
        # azione da eseguire: estrae un numero a caso: se questo è minore di epsilon allora fa azione casuale x=(0,q_t), altrimenti fa argmax_a(Q(s,a))
        #if state[0] <= 0:
        #    action = 0

        if np.random.rand() <= self.epsilon and state[1] < 4:#
            n = state[0]
            p = 1 / (self.time_subdivisions - state[1])
            action = np.random.binomial(n, p)
            action = round(np.linspace(0, self.inventory, self.inventory)[action])  #qui dovrebbe andare da -20 a + 20      

        elif state[1] == 4:
            action = state[0]
            #if state[0] <= 0:
            #    action = 0

        else:
            action = self.q_action(state, min_p, max_p)

        return action

    def reward(self, inv, x, data):
        '''
        calculates the reward of going slice and dice between the intervals with the quantity chosen to be traded within the intervals
        '''
        reward = 0
        inventory_left = inv
        M = len(data)
        xs = x/M
        for i in range(1, M):
            if i+1 < len(data):
                reward += inventory_left * (data[i-1] - data[i]) - self.a_penalty * (xs ** 2)

                inventory_left -= xs

        return reward

    def train_1(self, transitions, data, p_min, p_max):
        '''
        performs the training of the Q-Function approximated as a NN, manages the corner cases of interval = 4 or = 5
        '''
        #PROBLEMA E' QUI NEL SAMPLING E NELL'USO DI QUESTI SAMPLING

        state       = [tup[:2] for tup in transitions ]
        act         = [tup[2] for tup in transitions  ]
        next_state  = [tup[3:5] for tup in transitions]
        reward      = [tup[5] for tup in transitions  ]
        
        current_Q = self.eval_Q(state, act, p_min, p_max,'tensor', 'main')
        target = []

        self.matrix     [state[:][1][0], state[:][1][1]] +=  1
        #self.matrix_heat[state[:][1][0], state[:][1][1]] = act[??????] ###### che ci metto?????

        for (s, next_s, rew, ac) in zip(state, next_state, reward, act):

            #self.matrix_heat[s[0], s[1]] = act[ac]
            
            # qui aumenta il counter per gli states quando vengono sampled 

            if next_s[1] >= 4:
                target_value = rew
                target.append(target_value)

            elif next_s[1] == 3:
                q = s[0]
                x = ac
                future_next_price = data[0]
                next_state_price = data[-1]
                a = self.a_penalty
#
                best_future_action = self.q_action(next_s, p_min, p_max)
                correction_term = (q - x) * (future_next_price - next_state_price) - a * ((q - x) ** 2)
#
                target_value = rew + self.gamma * correction_term # questo è 0 se reward è 0 perchè gamma è 0
                target.append(target_value)

            else :
                best_future_action = self.q_action(next_s, p_min, p_max)
                target_value = rew + self.gamma * torch.max(self.eval_Q(next_s, best_future_action, p_min, p_max, 'scalar', 'target'))
                target.append(target_value)

        total_norm = 0
        for p in self.main_net.parameters():

            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2

            grad_norm = total_norm ** 0.5

        target = torch.tensor(target, dtype=torch.float32).reshape(-1,1)
        loss = F.mse_loss(target, current_Q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1)
        self.optimizer.step()

        if self.timestep % self.update_target_steps == 0:
            self._update_target_net()
            self.epsilon = self.epsilon * self.epsilon_decay
        
        torch.save(self.main_net.state_dict(), 'model.pth')

        return loss.cpu().item(), grad_norm, self.matrix, self.matrix_heat


    def test(self, inv, tempo, dati):
        '''
        does the testing on new data using the weights of the already trained NN
        '''
        model = self.main_net
        model.load_state_dict(torch.load('model.pth'))
        #torch.load_state()
        state = [inv, tempo, dati[0]]
        p_min = dati.min()
        p_max = dati.max()

        if tempo == 5:
            x = inv
        else:
            x = self.action(state, p_min, p_max)

        reward = self.reward(inv, x, dati)

        new_inv = inv - x
        #next_state = [new_inv,    tempo + 1,    dati[-2]]

        return (new_inv, x, reward)

    def PeL_QL(self, strategy, data):
        '''
        calculates the Profit and Loss of the strategy found by DQN
        '''
        PeL = 0
        a = self.a_penalty
        M = len(data)

        for i in range(self.time_subdivisions):

            x = self.lots_size * strategy[0]
            xs = x / M

            for t in range(M):
                if t + 1 < len(data):
                    PeL += xs * data[t] - a * (xs ** 2) 

        return np.double(PeL)


    def PeL_TWAP(self, data):
        '''
        Calculates the Profit and Loss of the TWAP strategy
        '''
        PeL = 0
        M = len(data)
        a = self.a_penalty

        x = self.inventory / self.time_subdivisions * self.lots_size
        xs = x / M

        for i in range(self.time_subdivisions):

            for t in range(M):
                if t+1 < len(data):
                    PeL += xs * data[t] - a * (xs ** 2)
        return PeL

    def step(self, inv, tempo, data):
        '''
        function that manages the states, performs the action and calculates the reward, in addition it fills up the replay buffer 
        '''
        #iter = 1
        self.timestep += 1
        state = [inv, tempo]
        #_dict.update({self.timestep : state, 'n': 0})
        p_min, p_max = data.min(), data.max()
        x = self.action(state, p_min, p_max)
        r = self.reward(inv, x, data)
        new_inv = inv - x
        next_state = [new_inv, tempo + 1]
        self.memory.add(state[0], state[1], x, next_state[0], next_state[1],  r)

        if len(self.memory) < self.batch_size:

            return 1, 0, np.zeros((21,5)), np.zeros((21,5)), new_inv, x, r

        else:

            transitions = self.memory.sample(self.batch_size)
        
        #salva i pesi qui?
        return *self.train_1(transitions, data, p_min, p_max), new_inv, x, r#, _dict


if __name__ == '__main__': 

    def sliceData(price, slici):

        step = int(len(price)/slici)
        y = np.zeros((slici,step))

        for i, ii in zip(range(slici), range(step, len(price), step)):
            it = step * i
            y[i, :] = price[it:ii]

        return y

    def doAve(a):
        aa = np.asarray(a)
        ai = aa.reshape(-1, 5)
        mean = ai.mean(axis = 0) 
        std = np.empty(5)
        for i in range(ai.shape[1]):
            std[i] = stdev(np.double(ai[:,i]))#/(ai.shape[0] - 1)
            
        return np.double(mean), np.double(std)

    def doTrain(age, numIt = 200):

        act_hist = []
        loss_hist = []
        rew_hist = []
        grad_hist = []
        matrix = np.zeros((21,5))

        data = Ambiente().abm(numIt=numIt) #------> questa la faccio da fuori? NO

        for j in tqdm(range(numIt)):
            slices = 5
            ss = sliceData(data[:,j], slices)
            inv = 20
            tempo = 0

            for i in tqdm(range(slices)):

                dati = ss[i,:] # considero slice da 5 (720 osservazioni alla volta)
                loss, grad, _dict, mat_act, new_inv, action, reward = age.step(inv, tempo, dati) 
                inv = new_inv 
                tempo += 1

                rew_hist.append(reward)
                act_hist.append(action)
                grad_hist.append(grad)
                loss_hist.append(loss)

                #matrix[new_inv, i] = np.asarray(act_hist)#.mean(axis = 0)

        act_mean, act_sd = doAve(act_hist)
        loss_mean, loss_sd = doAve(loss_hist)
        rew_mean, rew_sd = doAve(rew_hist)

        return (act_mean, act_sd, act_hist, loss_mean, loss_sd, rew_mean, rew_sd, loss_hist, rew_hist, _dict, matrix)

    def doTest(age, numIt = 100):
        data = Ambiente().abm(seed = 10, numIt=numIt)
        act = []
        re = []
        transaction_cost_balance = []
        mean_list = []
        for j in tqdm(range(numIt)):
            slices = 5
            ss = sliceData(data[:,j], slices)
            inv = 20
            tempo = 0

            for i in tqdm(range(slices)):

                
                dati = ss[i,:]
                selling_strategy = []#deque()
                (new_inv, x, reward) = age.test(inv, tempo, dati)
                tempo += 1
                inv = new_inv
                re.append(reward)
                act.append(x)
                selling_strategy.append(x)

                transaction_cost_balance.append((age.PeL_QL(selling_strategy, dati) - age.PeL_TWAP(dati)) / (age.PeL_TWAP(dati)))

        mean_list.append(transaction_cost_balance)
        #performance_list.append(performance(transaction_cost_balance))
        
        return doAve(mean_list), act, doAve(act), doAve(re)
    
    def get_heatmap(agent):

        def choose_best_action(q,t):   
            define_state = [q,t]
            return agent.action(define_state, 0, 1)

        array = np.zeros((21, 5))

        for q in range(21):
            for t in range(5):
                x = choose_best_action(q,t)
                array[q][t] = x

        return array

    def run(test = False):
        numIt = 10_000
        age = Agente(inventario = 20, numTrain = numIt)

        #for i in range(3):
        
        act_mean, act_sd, act_hist, loss_mean, loss_sd, rew_mean, rew_sd, loss_hist, rew_hist, _dict, matrix = doTrain(age, numIt)

        matrix_ = get_heatmap(age)

        #if test == True:
        #    pel, azioni, azioni_med, ricompensa = doTest(age, 1_000)
        #    return a, a_var, a_t, b, b_var, c, c_var, pel, azioni, azioni_med, ricompensa, loss_hist, rew_hist
        
        if test == False:

            return act_mean, act_sd, act_hist, loss_mean, loss_sd, rew_mean, rew_sd, loss_hist, rew_hist, _dict, matrix_
    
    
    start_time = time.time()

    act_mean, act_sd, act_hist, loss_mean, loss_sd, rew_mean, rew_sd, loss_hist, rew_hist, _dict, matrix = run(False)

    print('average action chosen from train =', act_mean, ',actions train sd = ', act_sd ,
    ', reward train =' ,rew_mean ,',reward sd = ', rew_sd , ',average loss from NN =', loss_mean, ',loss sd =', loss_sd,
    '\n', "--- %s minutes ---" % ((time.time() - start_time)/60), '\n','states=', _dict, '\n', 'act=', matrix)#
    #'\n',',PeL performance=', pel,',average actions test=', azioni_med, ',ricompensa = ', ricompensa, 
    #'\n', ',last train=', a_t[-5:], ' last test = ', azioni[-5:])

    plt.plot(loss_hist)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()
    plt.plot(rew_hist)
    plt.xlabel('iterations')
    plt.ylabel('rewards')
    plt.show()
    plt.hist(act_mean)
    plt.title('average action train')
    plt.show()
    sns.heatmap(np.asarray(_dict))
    plt.show()
    sns.heatmap(np.asarray(matrix))
    plt.show()
