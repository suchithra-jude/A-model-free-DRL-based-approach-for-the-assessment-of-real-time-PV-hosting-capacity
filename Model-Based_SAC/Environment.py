import sys
sys.path.append(r'C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10')
import powerfactory
import numpy as np
import pandas as pd
import math
import random

LoadShape = np.loadtxt("load_profile_1.txt", dtype=float)
PVShape = np.loadtxt("PVloadshape.txt", dtype=float)

class EnviroNet(object):
    def __init__(self):
        self.app = powerfactory.GetApplication()
        self.app.ActivateProject("Net_28_Clean.IntPrj")
        # self.PVsystems = self.app.GetCalcRelevantObjects('*.ElmPvsys')
        # self.Terms = self.app.GetCalcRelevantObjects('TES_*.ElmTerm')
        self.PVsystems = sorted(self.app.GetCalcRelevantObjects('*.ElmPvsys'), key=lambda obj: obj.GetAttribute('loc_name'))
        self.Terms = sorted(self.app.GetCalcRelevantObjects('TES_*.ElmTerm'), key=lambda obj: obj.GetAttribute('loc_name'))
        self.duration = len(PVShape)

        self.action_space = []
        for PV in self.PVsystems:
            self.action_space.append(PV)

    def state_shape(self):
        return ((len(self.Terms)*2), )

    def state_size(self):
        return (len(self.Terms)*2)

    def action_size(self):
        return len(self.action_space)

    def initialize(self, time):
        Solar_Irrad = PVShape[time]

        Loads = self.app.GetCalcRelevantObjects('*.ElmLodlv')
        for Load in Loads:
            Load.slini = LoadShape[time]
            Load.coslini = 0.95

        for idx, PV in enumerate(self.PVsystems):
            PV.sgn = 4
            PV.pgini = 0
            PV.qgini = 0

        # calculate loadflow
        ldf = self.app.GetFromStudyCase('ComLdf')
        ldf.iopt_net = 1
        ldf.Execute()

    def step_env(self, time):
        Solar_Irrad = PVShape[time]
        if Solar_Irrad > 0.75 and time % 3 == 0:
            Solar_Irrad = 1

        Loads = self.app.GetCalcRelevantObjects('*.ElmLodlv')
        for Load in Loads:
            Load.slini = LoadShape[time]
            Load.coslini = 0.95

        for idx, PV in enumerate(self.PVsystems):
            PV.sgn = 4
            PV.pgini = 0
            PV.qgini = 0

        # calculate loadflow
        ldf = self.app.GetFromStudyCase('ComLdf')
        ldf.iopt_net = 1
        error = ldf.Execute()
        done = False
        if error == 1:
            done = True
            return [[], done]

        CustomerVoltage = np.zeros(len(self.Terms), dtype=float)
        index = 0
        for Term in self.Terms:
            CustomerVoltage[index] = Term.GetAttribute('m:U1')
            index += 1
        Current_state = np.append(CustomerVoltage, np.full(len(self.Terms), Solar_Irrad))

        return [Current_state, done]


    def step_ctr(self, time, actions):
        Solar_Irrad = PVShape[time]
        PV_Ratting_Array = np.empty(len(self.PVsystems))
        Total_Poutput = np.empty(len(self.PVsystems))
        Max_HC = 50
        Min_HC = 20

        Correct_actions = actions
        actions = ((actions + 1) / 2)
        Meanactions = np.mean(actions)
        actions = np.clip(actions, 0.9 * Meanactions, 1.1 * Meanactions)
        actions = np.clip(actions, 0, 1)

        for index in range(0, len(self.PVsystems)):
            PV_Rating = Min_HC + ((Max_HC - Min_HC) * actions[index])
            if Solar_Irrad < 0.01:
                PV_Rating = 0
                PV_Ratting_Array[index] = PV_Rating
                Total_Poutput[index] = PV_Rating*Solar_Irrad
                self.action_space[index].sgn = Min_HC
                self.action_space[index].pgini = Total_Poutput[index]
            else:
                PV_Ratting_Array[index] = PV_Rating
                Total_Poutput[index] = PV_Rating*Solar_Irrad
                self.action_space[index].sgn = PV_Rating
                self.action_space[index].pgini = Total_Poutput[index]


        ldf = self.app.GetFromStudyCase('ComLdf')
        ldf.iopt_net = 1
        error = ldf.Execute()
        if error == 1:
            done = True
            Vmax_limit = 258
            Vmin_limit = 220
            Nominal_Voltage = 230
            reward = -2 * (abs(Vmax_limit - Nominal_Voltage) + abs(Vmin_limit - Nominal_Voltage)) / 0.05
            return [[], reward, [], done, [], [], [], [], [], [], []]

        CustomerVoltage = np.zeros(len(self.Terms), dtype=float)
        for index, Term in enumerate(self.Terms):
            CustomerVoltage[index] = Term.GetAttribute('m:U1')
        MaxVoltage = 1000*np.nanmax(CustomerVoltage)
        MinVolatge = 1000*np.nanmin(CustomerVoltage)
        Next_state = np.append(CustomerVoltage, np.full(len(self.Terms), Solar_Irrad))

        done = False
        Vmax_limit = 258
        Vmin_limit = 207
        Nominal_Voltage = 230
        penalty = (-10*Max_HC*len(self.PVsystems)) - 10*(abs(MaxVoltage - Nominal_Voltage)) - 10*abs(MinVolatge - Nominal_Voltage)

        reward = -((Max_HC*len(self.PVsystems))+1 - np.sum(PV_Ratting_Array))

        if (MaxVoltage > Vmax_limit) or (MinVolatge < Vmin_limit):
            reward = penalty

        if (Solar_Irrad == 0):
            reward = 0

        return [Next_state, reward, Correct_actions, done, MaxVoltage, MinVolatge, CustomerVoltage, Total_Poutput, PV_Ratting_Array, Solar_Irrad]


















