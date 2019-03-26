import tensorflow as tf
import copy
import numpy as np
import copy



class Agent():

    def __init__(self,obsize,actsize,sess,gamma,epsilon = None,plus = True):
        state = tf.placeholder(tf.float32,[None,obsize])
        # actions = tf.placeholder(tf.int32, [None])
        if plus == True:
            if epsilon != None:
                w1 = tf.placeholder(tf.float32,[obsize,2 * obsize]) + epsilon[0]
                b1 = tf.placeholder(tf.float32,[1,2 * obsize]) + epsilon[1]
                x_w1_b1 = tf.nn.tanh(tf.matmul(state,w1) + b1)

                w2 = tf.placeholder(tf.float32,[2 * obsize,int(0.5 * obsize)]) + epsilon[2]
                b2 = tf.placeholder(tf.float32,[1,int(0.5 * obsize)]) + epsilon[3]
                x_w2_b2 = tf.nn.tanh(tf.matmul(x_w1_b1,w2) + b2)

                w3 = tf.placeholder(tf.float32,[int(0.5 * obsize),actsize]) + epsilon[4]
                b3 = tf.placeholder(tf.float32,[1,actsize])+ epsilon[5]
                action_vec = tf.matmul(x_w2_b2,w3) + b3

                action = tf.math.argmax(action_vec,axis = 1)

                theta = [w1,b1,w2,b2,w3,b3]
                state_theta = [w1,b1,w2,b2,w3,b3]
                state_theta.append(state)

            else:
                w1 = tf.placeholder(tf.float32,[obsize,2 * obsize]) 
                b1 = tf.placeholder(tf.float32,[1,2 * obsize]) 
                x_w1_b1 = tf.nn.tanh(tf.matmul(state,w1) + b1)

                w2 = tf.placeholder(tf.float32,[2 * obsize,int(0.5 * obsize)]) 
                b2 = tf.placeholder(tf.float32,[1,int(0.5 * obsize)]) 
                x_w2_b2 = tf.nn.tanh(tf.matmul(x_w1_b1,w2) + b2)

                w3 = tf.placeholder(tf.float32,[int(0.5 * obsize),actsize]) 
                b3 = tf.placeholder(tf.float32,[1,actsize]) 

                action_vec = tf.matmul(x_w2_b2,w3) + b3

                # action = tf.math.argmax(action_vec,axis = 1)

                theta = [w1,b1,w2,b2,w3,b3]
                state_theta = [w1,b1,w2,b2,w3,b3]
                state_theta.append(state)

        if plus == False:
            if epsilon != None:
                w1 = tf.placeholder(tf.float32,[obsize,2 * obsize]) - epsilon[0]
                b1 = tf.placeholder(tf.float32,[1,2 * obsize]) - epsilon[1]
                x_w1_b1 = tf.nn.tanh(tf.matmul(state,w1) + b1)

                w2 = tf.placeholder(tf.float32,[2 * obsize,int(0.5 * obsize)]) - epsilon[2]
                b2 = tf.placeholder(tf.float32,[1,int(0.5 * obsize)]) - epsilon[3]
                x_w2_b2 = tf.nn.tanh(tf.matmul(x_w1_b1,w2) + b2)

                w3 = tf.placeholder(tf.float32,[int(0.5 * obsize),actsize]) - epsilon[4]
                b3 = tf.placeholder(tf.float32,[1,actsize])- epsilon[5]
                action_vec = tf.matmul(x_w2_b2,w3) + b3

                action = tf.math.argmax(action_vec,axis = 1)

                theta = [w1,b1,w2,b2,w3,b3]
                state_theta = [w1,b1,w2,b2,w3,b3]
                state_theta.append(state)

            else:
                w1 = tf.placeholder(tf.float32,[obsize,1 * obsize]) 
                b1 = tf.placeholder(tf.float32,[1,1 * obsize]) 
                x_w1_b1 = tf.nn.tanh(tf.matmul(state,w1) + b1)

                w2 = tf.placeholder(tf.float32,[1 * obsize,int(0.5 * obsize)]) 
                b2 = tf.placeholder(tf.float32,[1,int(0.5 * obsize)]) 
                x_w2_b2 = tf.nn.tanh(tf.matmul(x_w1_b1,w2) + b2)

                w3 = tf.placeholder(tf.float32,[int(0.5 * obsize),actsize]) 
                b3 = tf.placeholder(tf.float32,[1,actsize]) 

                action_vec = tf.matmul(x_w2_b2,w3) + b3

                # action = tf.math.argmax(action_vec,axis = 1)

                theta = [w1,b1,w2,b2,w3,b3]
                state_theta = [w1,b1,w2,b2,w3,b3]
                state_theta.append(state)


        self.sess = sess
        self.state_theta = state_theta
        self.gamma = gamma
        self.action_vec = action_vec
        # self.action = action
        self.theta_shape = [item.shape for item in theta]
        self.len_theta = [item.shape[0] * item.shape[1] for item in theta]
        self.len_theta_total = sum(self.len_theta)

    def _take_action(self,info):
        # print(len(info))
        info = [(self.state_theta[i],info[i]) for i in range(len(info))]
        # for i in range(len(info)):
        #     print(info[i][0],info[i][1].shape)
        return self.sess.run([self.action_vec],feed_dict = {k:v for (k,v) in info})

    def _roll_out(self,env,theta,render = False):
        reward = 0
        d = False
        obs = env.reset()
        iteration = 0
        # theta_state = theta
        theta_state = copy.deepcopy(theta)
        while not d:
            if render:
                env.render()
            if iteration == 0:
                theta_state.append(obs.reshape(1,len(obs)))
            else:
                theta_state[-1] = obs.reshape(1,len(obs))
            a = self._take_action(theta_state)[0]
            a = np.exp(a)/np.sum(np.exp(a),keepdims = True)
            # print(a[0])
            # print(np.random.choice(2, 1, p=a[0]))
            # obs,r,d, _ = env.step(a[0])
            obs,r,d, _ = env.step(np.random.choice(2, 1, p=a[0])[0])

            reward += r 
            # * (self.gamma ** iteration)
            iteration +=1
        env.close()
        self.reward = reward
        # print(reward)

        return reward


        

