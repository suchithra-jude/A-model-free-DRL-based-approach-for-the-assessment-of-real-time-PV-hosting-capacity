import numpy as np
import pandas as pd
import datetime
from sac import Agent
from Environment import EnviroNet
from utils import plot_result_curve

if __name__ == '__main__':
    Date_time = np.loadtxt("Datetime.txt", dtype=str)
    Episodes = 1
    reward = 0
    Vmax = 0
    Vmin = 0
    Term_Voltage = 0
    P_Output = 0
    PV_Rating = 0
    Irradaition = 0

    env = EnviroNet()
    Time_steps = env.duration
    agent = Agent(input_dims=env.state_shape(), n_actions=env.action_size(), episodes=Episodes)

    load_checkpoint = True

    evaluate = False
    if load_checkpoint:
        n_steps = 0
        Time = 0
        agent.batch_size = 50
        while n_steps <= agent.batch_size + 6:
            observation = env.step_env(Time)
            action = np.random.rand(env.action_size())
            observation_, reward, correctAction, done, Vmax, Vmin, Term_Voltage, P_Output, PV_Rating, Irradaition = env.step_ctr(Time, action)
            agent.remember(observation, correctAction, reward, observation_, done)
            n_steps += 1
            agent.learn()
        agent.load_models()
        evaluate = True


    score_history = []
    Max_V_history = []
    Voltage_history = np.zeros([Time_steps, env.PVsystems])
    PV_Output_history = np.zeros([Time_steps, env.PVsystems])
    PV_Rating_history = np.zeros([Time_steps, env.PVsystems])
    Reward_history = np.zeros([48, Episodes])
    Reward_history_plot = np.zeros([int(Episodes*(Time_steps/48)), 2])

    figure_ResultR = 'plots/CumulativeReward.png'
    figure_ResultV = 'plots/Voltage.png'

    best_score = -30000
    avg_score = -30000
    score = 0

    for Time in range(0, Time_steps):
        done = False
        observation = env.step_env(Time)

        for Episode in range(0, Episodes):
            action = agent.choose_action(observation)
            out1, out2, out3, out4, out5, out6, out7, out8, out9, out10 = env.step_ctr(Time, action)
            score += out2
            done = out4
            if not done:
                observation_ = out1
                reward = out2
                correctAction = out3
                done = out4
                Vmax = out5
                Vmin = out6
                Term_Voltage = out7
                P_Output = out8
                PV_Rating = out9
                Irradaition = out10
                if (not load_checkpoint) and Irradaition > 0:
                    agent.remember(observation, correctAction, reward, observation_, done)
                    agent.learn()
                observation = observation_

            Reward_history[Time % 48, Episode] = reward

        if (not load_checkpoint) and (Time == 2500):
            agent.entropy_alpha = 0.2

        if (Time+1) % 48 == 0:
            Daily_reward = np.sum(Reward_history, axis=0)
            for Episode in range(0, Episodes):
                Reward_history_plot[(Episodes * (int((Time+1) / 48) - 1)) + Episode, 0] = (Time+1)/48
                Reward_history_plot[(Episodes * (int((Time+1) / 48) - 1)) + Episode, 1] = Daily_reward[Episode]

            score_history.append(score/Episodes)
            avg_score = np.mean(score_history[-5:])
            Reward_history = np.zeros([48, Episodes])
            score = 0

        Max_V_history.append(Vmax)
        Voltage_history[Time, :] = Term_Voltage
        PV_Output_history[Time, :] = P_Output
        PV_Rating_history[Time, :] = PV_Rating

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('Time:', Date_time[Time], 'score: %.1f' % (score/Episodes),
            'best score: %.1f' % best_score, 'maximum voltage: %.4f' % Vmax, 'minimum voltage: %.4f' % Vmin,
            'avg_P_Out: %.4f' % (np.mean(P_Output)),  'PV_Rating: %.4f' % (np.mean(PV_Rating)), 'Irradiation: %.4f' % Irradaition)

    if not load_checkpoint:
        agent.save_models()
        DF = pd.DataFrame(Reward_history_plot)
        DF.to_csv("PV Ratings/Reward_history_plot.csv")

    if evaluate:
        DF = pd.DataFrame(Voltage_history)
        DF.to_csv("PV Ratings/SAC_Voltage_history.csv")
        DF = pd.DataFrame(PV_Output_history)
        DF.to_csv("PV Ratings/PV_Output.csv")
        DF = pd.DataFrame(PV_Rating_history)
        DF.to_csv("PV Ratings/PV_Rating_history.csv")

    x = [i + 1 for i in range(len(score_history))]
    y = [i + 1 for i in range(len(Max_V_history))]
    plot_result_curve(x, score_history, figure_ResultR, 'Cumulative reward')
    plot_result_curve(y, Max_V_history, figure_ResultV, 'Maximum Voltage')