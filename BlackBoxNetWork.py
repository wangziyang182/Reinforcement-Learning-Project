from __future__ import division
import tensorflow as tf
import numpy as np
import collections


class BlackBox:
    
    def __init__(self,action_space,state_space,sess,continuous_action = False):
        self.sess = sess
        self.action_space = action_space
        self.Weight = []
        self.Bias = []
        self.shape = []
        self._init_model(action_space,state_space,continuous_action = False)
        self.sess.run(tf.global_variables_initializer())   
        self.saver = tf.train.Saver()
 

    
    def _init_model(self,action_space,state_space,continuous_action = False):
        state = tf.placeholder(tf.float32,[None,state_space])
        x_w_b1 = self._add_layer(state,state_space,state_space * 2,activation = tf.nn.relu)
        x_w_b2 = self._add_layer(x_w_b1,state_space * 2,int(state_space / 2),activation = tf.nn.relu)
        if continuous_action == False:
            action = self._add_layer(x_w_b2,int(state_space / 2),action_space, activation = tf.nn.softmax)
            self.continuous = False
        else:
            action = self._add_layer(x_w_b2,int(state_space / 2),action_space, activation = None)
            self.continuous = True

        self.state = state
        self.action = action

    def _add_layer(self,x,in_size, out_size,activation = None):
        w = tf.Variable(tf.random.truncated_normal([in_size,out_size]),tf.float32)
        b = tf.Variable(tf.random.truncated_normal([1,out_size]),tf.float32)
        Xw_b = tf.matmul(x,w) + b

        self.Weight.append(w)
        self.Bias.append(b)

        if activation == None:
            return x_w_b
        else:
            return activation(Xw_b)

    def update_shape(self):
        self.shape = [(list(self.Weight[i].get_shape()),list(self.Bias[i].get_shape())) for i in range(len(self.Weight))]

    def get_param(self):
        param = [(self.Weight[i].eval(self.sess),self.Bias[i].eval(self.sess)) for i in range(len(self.Weight))]
        return param

    def set_param(self,param):
        for i in range(len(param)):
            self.Weight[i].load(param[i][0],self.sess)
            self.Bias[i].load(param[i][1],self.sess)

    def get_flat_param(self):
        params = [np.concatenate((self.Weight[i].eval(self.sess).flatten(),self.Bias[i].eval(self.sess).flatten())) for i in range(len(self.Weight))]
        param_flat = [item for sublist in params for item in sublist]

        return param_flat

   
    def set_flat_param(self,param):
        index = 0
        param = param.flatten()
        for i in range(len(self.shape)):
            w_len = self.shape[i][0][0] * self.shape[i][0][1]
            self.Weight[i].load(param[index:index + w_len].reshape(self.shape[i][0]),self.sess)
            index = index + w_len
            b_len = self.shape[i][1][0] * self.shape[i][1][1]
            self.Bias[i].load(param[index:index + b_len].reshape(self.shape[i][1]),self.sess)
            index = index + b_len


    def _take_action(self,state,greedy = True):
        if self.continuous == True:
            return self.sess.run(self.action, feed_dict = {self.state: state}).flatten()

        else:
            if greedy == True:
                return np.argmax(self.sess.run(self.action, feed_dict = {self.state: state}).flatten())
            else:
                action = self.sess.run(self.action, feed_dict = {self.state: state}).flatten()
                return np.random.choice(self.action_space, 1, p=action)

    def roll_out(self,param,env,greedy = True,render = False):
        #pass in parameter to function
        self.set_flat_param(param)

        #inital simulation
        done = False
        state = env.reset()
        state = state[np.newaxis,...]
        individual_fit = 0


        while not done:
            #render the env
            if render == True:
                env.render()
            else:
                pass

            action = self._take_action(state,greedy)
            state,reward,done, _ = env.step(action)
            state = state[np.newaxis,...]
            individual_fit += reward

        env.close()
        return individual_fit








        




