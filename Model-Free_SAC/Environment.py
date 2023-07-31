from SurrogateModel import SurrogateAgent
import numpy as np
import pandas as pd
import math
import random


LoadShape = np.loadtxt("load_profile_1.txt", dtype=float)
PVShape = np.loadtxt("PVloadshape.txt", dtype=float)
Test_target_Buffer = pd.read_csv('PV Ratings/Test_target_Buffer.csv')
Test_target_Buffer = Test_target_Buffer.values
Test_target_Buffer = Test_target_Buffer[:, 1:]
Test_observation_Buffer = pd.read_csv('PV Ratings/Test_observation_Buffer.csv')
Test_observation_Buffer = Test_observation_Buffer.values
Test_observation_Buffer = Test_observation_Buffer[:, 1:]

class EnviroNet(object):
    def __init__(self):
        self.duration = len(PVShape)
        self.PVsystems = 28
        batch_size = 46
        self.NNagent = SurrogateAgent(n_actions=self.PVsystems)
        for Episode in range(0, len(Test_target_Buffer)):
            batch = np.random.choice(len(Test_target_Buffer), batch_size)
            observation = Test_observation_Buffer[batch]
            target = Test_target_Buffer[batch]
            self.NNagent.learn(observation, target)
        self.NNagent.load_models()

    def state_shape(self):
        return ((self.PVsystems*2), )

    def state_size(self):
        return (self.PVsystems*2)

    def action_size(self):
        return self.PVsystems


    def step_env(self, time):
        Solar_Irrad = PVShape[time]
        if Solar_Irrad > 0.75 and time % 3 == 0:
            Solar_Irrad = 1

        Current_state = np.zeros(self.PVsystems*3, dtype=float)
        index = 0
        for _ in range(0, self.PVsystems):
            Pgen = 0
            Pload = LoadShape[time]
            Qload = Pload*math.tan(math.acos(0.95))
            Current_state[index] = Pgen
            Current_state[index+1] = Pload
            Current_state[index+2] = Qload
            index += 3

        CustomerVoltage = self.NNagent.choose_action(Current_state)
        Current_state = np.append(CustomerVoltage, np.full(self.PVsystems, Solar_Irrad))

        return Current_state


    def step_ctr(self, time, actions):
        Solar_Irrad = PVShape[time]
        PV_Ratting_Array = np.empty(self.PVsystems)
        Total_Poutput = np.empty(self.PVsystems)
        Max_HC = 50
        Min_HC = 20

        Correct_actions = actions
        actions = ((actions + 1) / 2)
        Meanactions = np.mean(actions)
        actions = np.clip(actions, 0.9 * Meanactions, 1.1 * Meanactions)
        actions = np.clip(actions, 0, 1)

        for index in range(0, self.PVsystems):
            PV_Rating = Min_HC + ((Max_HC - Min_HC) * actions[index])
            if Solar_Irrad < 0.01:
                PV_Rating = 0
            PV_Ratting_Array[index] = PV_Rating
            Total_Poutput[index] = PV_Rating*Solar_Irrad

        Current_state = np.zeros(self.PVsystems*3, dtype=float)
        index = 0
        for idx in range(0, self.PVsystems):
            Pgen = Total_Poutput[idx]
            Pload = LoadShape[time]
            Qload = Pload*math.tan(math.acos(0.95))
            Current_state[index] = Pgen
            Current_state[index+1] = Pload
            Current_state[index+2] = Qload
            index += 3

        CustomerVoltage = self.NNagent.choose_action(Current_state)
        MaxVoltage = 1000*np.nanmax(CustomerVoltage)
        MinVolatge = 1000*np.nanmin(CustomerVoltage)
        Next_state = np.append(CustomerVoltage, np.full(self.PVsystems, Solar_Irrad))

        done = False
        Vmax_limit = 258
        Vmin_limit = 207
        Nominal_Voltage = 230
        penalty = (-10*Max_HC*self.PVsystems) - 10*(abs(MaxVoltage - Nominal_Voltage)) - 10*abs(MinVolatge - Nominal_Voltage)

        reward = -((Max_HC*self.PVsystems)+1 - np.sum(PV_Ratting_Array))

        if (MaxVoltage > Vmax_limit) or (MinVolatge < Vmin_limit):
            reward = penalty

        if (Solar_Irrad == 0):
            reward = 0

        return [Next_state, reward, Correct_actions, done, MaxVoltage, MinVolatge, CustomerVoltage, Total_Poutput, PV_Ratting_Array, Solar_Irrad]


















