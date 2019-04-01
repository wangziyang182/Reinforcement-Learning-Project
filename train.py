import gym
import numpy as np
import tensorflow as tf
from BlackBoxNetWork import BlackBox
import os
import matplotlib.pyplot as plt


class train_ES():
    def __init__(
                self,
                iterations = 200,
                num_perturbations = 200,
                # env = 'BipedalWalker-v2',
                env = 'CartPole-v1',
                gamma = 0.99,
                sigma = 3,
                lr = 1 * 1e-2,
                max_length = 2000,
                num_test = 1,
                continuous_action = True
                ):
        self.iterations = iterations
        self.num_perturbations = num_perturbations
        self.env = env
        # envname = 'CartPole-v1'
        self.gamma = gamma
        self.sigma = sigma
        self.lr = lr
        self.max_length = max_length
        self.num_test = num_test
        self.continuous_action = continuous_action

    def train(self):
        np.random.seed(0)
        for test in range(self.num_test):
            fit_list = []
            iteration_list = []

            plt.ion()
            plt.show()

            #set up environment
            env = gym.make(self.env)
            # action_space = env.action_space.shape[0]
            action_space = env.action_space.n
            state_space = env.observation_space.low.size

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            #set up function mapping
            bbnw = BlackBox(action_space,state_space,sess,continuous_action = self.continuous_action)
            bbnw.update_shape()

            #get param for the len
            param = np.array(bbnw.get_flat_param())
            for iteration in range(self.iterations):
                print('iteration',iteration)
                #gaussian Matrices
                noises = self.sigma * np.random.randn(self.num_perturbations,len(param))
                noisy_param = param + noises
                #get the fittness score for number of pertubations
                fittness = []
                i = 0
                for ind in noisy_param:

                    if i % 200 == 0:
                        render = True
                    else:
                        render = False

                    #do the roll out
                    ind_fit = bbnw.roll_out(ind,env,render = render)
                    fittness.append(ind_fit)
                    i += 1
                pert_fittness = np.array(fittness).reshape((len(fittness),1))

                #record average reward
                fit_list.append(np.sum(pert_fittness)/self.num_perturbations)
                iteration_list.append(iteration)
                plt.plot(iteration_list,fit_list,'r')
                plt.draw()
                plt.pause(0.3)

                #update param
                param = param + (self.lr / self.num_perturbations / self.sigma * (noisy_param.T@pert_fittness)).flatten()
                print("-" * 100)

            #save weights
            # bbnw.saver.save(bbnw.sess,)


    def save_param(self,BBNW,path,name):
        # BBNW.saver.save(bbnw.sess,)
        pass




if __name__ == '__main__':

    ES = train_ES(continuous_action = False)
    ES.train()

