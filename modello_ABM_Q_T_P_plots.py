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

in_features = 4


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

    def __init__(self, S0 = 10, mu = 0, kappa = 5, theta = 1, sigma = 0.01, lambd = 0.1, t0 = 0, t = 1, T = 3600, inv = 20): #T = 1.0, M = 7200, I = 1_000
        
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

    def normalise(self, inventory, time, price, x, min_p, max_p):
        '''
        performs the normalisation in the range [-1,1] for the feature of the NN
        '''
        q, x = self.inventory_action_transform(inventory, x)
        t = self.time_transform(time)
        p = self.price_normalise(price, min_p, max_p)
        return q, t, p, x

class ReplayMemory():
    '''
    Experience replay memory
    '''

    def __init__(self, capacity):

        self.memory = deque([], maxlen=capacity)

    def add(self, inv, time, price, x, next_inv, next_time, next_price, reward): #inv, time, price, var, x, next_state, reward state, action, next_state, reward
        
        self.memory.append([inv, time, price, x, next_inv, next_time, next_price, reward])

    def sample(self, batch_size):
        
        return rnd.sample(self.memory, batch_size)

    def __len__(self):
        
        return len(self.memory)

class Agente():

    def __init__(self, inventario, numTrain):

        self.train = numTrain
        self.memory = ReplayMemory(7_000)   
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
        self.update_target_steps = 1000
        self.lots_size = 100

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

            q, t, p, x = Ambiente().normalise(state[0], state[1], state[2], act, p_min, p_max)
            in_features = torch.tensor([q, t , p, x], dtype=torch.float)

        if type == 'tensor':

            features = []

            for i in range(len(state)):

                q, t, p, x = Ambiente().normalise(state[i][0], state[i][1], state[i][2], act[i], p_min, p_max)
                features.append(torch.tensor([q, t , p, x], dtype=torch.float))

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

            for i in range(int(state[0] + 1)):

                q, t , p, x = Ambiente().normalise(state[0], state[1], state[2], i, min_p, max_p) 
                features.append(torch.tensor([q, t , p, x], dtype=torch.float))

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
        if state[0] <= 0:
            action = torch.tensor([0.0] , dtype=torch.float)

        elif np.random.rand() <= self.epsilon and state[1] < 4:#
            n = state[0]
            p = 1/(self.time_subdivisions - state[1])
            action = np.random.binomial(n, p)
            action = round(np.linspace(0, self.inventory, self.inventory)[action])        
            
        elif state[1] >= 4:
            action = state[0]

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

        state       = [tup[:3] for tup in transitions ]
        act         = [tup[3] for tup in transitions  ]
        next_state  = [tup[4:7] for tup in transitions]
        reward      = [tup[7] for tup in transitions  ]

        current_Q = self.eval_Q(state, act, p_min, p_max,'tensor', 'main')
        target = []

        for (s, next_s, rew, ac) in zip(state, next_state, reward, act):

            if next_s[1] == 3:
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

            elif next_s[1] == 4:
                target_value = rew
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

        return loss.cpu().item(), grad_norm


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
        state = [inv, tempo , data[0]]
        p_min, p_max = data.min(), data.max()
        x = self.action(state, p_min, p_max)
        r = self.reward(inv, x, data)
        new_inv = inv - x
        next_state = [new_inv, tempo + 1 , data[-2]]
        self.memory.add(state[0], state[1], state[2], x, next_state[0], next_state[1], next_state[2], r)

        if len(self.memory) < self.batch_size:

            return 1, 0 , new_inv, x, r

        else:

            transitions = self.memory.sample(self.batch_size)
        
        #salva i pesi qui?
        return *self.train_1(transitions, data, p_min, p_max), new_inv, x, r


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
        ai = aa.reshape(-1,5)
        mean=ai.mean(axis=0)#np.empty((ai.shape[0]))
        #for i in range(5):
        #    mean[i] = np.mean(np.double(ai[i,:]))
        return np.double(mean)#.flatten()

    def doTrain(age, numIt = 200):

        act_hist = []
        loss_hist = []
        rew_hist = []
        grad_hist = []

        data = Ambiente().abm(numIt=numIt) #------> questa la faccio da fuori? NO

        for j in tqdm(range(numIt)):
            slices = 5
            ss = sliceData(data[:,j], slices)
            inv = 20
            tempo = 0

            for i in tqdm(range(slices)):

                dati = ss[i,:] # considero slice da 5 (720 osservazioni alla volta)
                loss, grad, new_inv, action, reward = age.step(inv, tempo, dati) 
                inv = new_inv 
                tempo += 1
                rew_hist.append(reward)
                act_hist.append(action)
                grad_hist.append(grad)
                loss_hist.append(loss)
        

        return (doAve(act_hist), act_hist, doAve(loss_hist),  doAve(rew_hist), loss_hist, rew_hist)

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
    
    def run():
        numIt = 2_000
        age = Agente(inventario = 20, numTrain = numIt)
        
        a, a_t, b, c , loss_hist, rew_hist = doTrain(age, numIt)
        pel, azioni, azioni_med, ricompensa = doTest(age, 1_000)

        return a, a_t,  b, c, pel, azioni, azioni_med, ricompensa, loss_hist, rew_hist
    
    
    start_time = time.time()

    a, a_t,  b, c, pel, azioni, azioni_med, ricompensa, loss_hist, rew_hist = run()

    print('average action chosen from train =', a, ', reward train =' , c, ',average loss from NN =', b, "--- %s seconds ---" % (time.time() - start_time))#
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
    plt.hist(a)
    plt.title('average action train')
    plt.show()