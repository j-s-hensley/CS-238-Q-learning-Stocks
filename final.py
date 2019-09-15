import numpy as np
import numpy.random as random
import sys
import collections
import operator
import matplotlib.pyplot as plt


def random_actions(len_test):
    """
    Generate a policy that acts uniformly at random, subject to only being able
    to sell shares when the number of shares owned is greater than zero.
    Inputs :
        len_test (int) : number of time steps to generate actions for
    Returns :
        pi (dict) : a policy for buying/selling/holding shares, where the keys
                    correspond to the stock price on a given day, and the values
                    correspond to the action for the i^{th} day
    """
    pi = {}
    current_stock = 0
    for i in range(len_test):
        if current_stock == 0:
            pi[i] = random.randint(2)
        else:
            pi[i] = random.randint(3)
        if pi[i] == 1:
            current_stock += 1
        if pi[i] == 2:
            current_stock -= 1
    return pi


def evaluate(pi,test_data):
    """
    Evaluate the reward earned by the policy pi on test_data.
    NOTE: This treats purchasing stock as a negative reward, thus discouraging
    any stock purchases. This is undesirable, so instead we use evaluate_2().
    Inputs :
        pi (dict) : a policy for buying/selling/holding shares, where the keys
                    correspond to the day, and the values correspond to the
                    action for the i^{th} day
        test_data (np.array) : an array with the stock prices from the test data
                               set and the percent change from the previous day,
                               from oldest to newest
    Returns :
        r (float) : the net gain in assets from buying and selling
    """
    r = 0
    current_stock = 0
    for i in range(test_data.shape[0]):
        cost = test_data[i,0]
        if pi[i] == 1:
            r -= cost
            current_stock += 1
        if pi[i] == 2:
            r += cost
            current_stock -= 1
    if current_stock > 0:
        r += current_stock*cost
    return r


def evaluate_2(pi,test_data,return_plot=False):
    """
    Evaluate the reward earned by the policy pi on test_data.
    NOTE : This calculates the exact same quantity as evaluate(), but does not
    treat purchasing stock as a penalty like evaluate() does, which is desirable
    for training.
    Inputs :
        pi (dict) : a policy for buying/selling/holding shares, where the keys
                    correspond to the day, and the values correspond to the
                    action for the i^{th} day
        test_data (np.array) : an array with the stock prices from the test data
                               set, from oldest to newest
    Returns :
        r (float) : the net gain in assets from buying and selling
    """
    r = 0
    current_stock = 0
    stock_owned = [0]
    for i in range(test_data.shape[0]-1):
        percent_change = test_data[i,1]
        if i > 0:
            r += (percent_change/1000000)*current_stock*test_data[i-1,0]
        state = int(percent_change+current_stock)
        if pi[state] == 1:
            current_stock += 1
        if pi[state] == 2:
            current_stock -= 1
        stock_owned.append(current_stock)
    if return_plot:
        return r, stock_owned
    else:
        return r


def preprocess(train,test,max_action):
    """
    Preprocess the input csv files.
    Inputs :
        train (str) : the filename of the training data, formatted as a csv
                      with closing prices and percent change from the previous
                      day on each row, with each new row being +1 day
        test (str) : the filename of the test data, formatted as a csv
                     with closing prices and percent change from the previous
                     day on each row, with each new row being +1 day
        max_action (int) : the number of possible actions to take (e.g. 3 for
                           buy, sell, and hold)
    Returns :
        data (np.array) : the training data as an array
        test_data (np.array) : the test data as an array
        bins (np.array) : the range of daily percent change values possible
    """
    data = np.loadtxt(train, dtype=float,delimiter=',')
    test_data = np.loadtxt(test, dtype=float,delimiter=',')

    for i in range(data.shape[0]):
        data[i,1] = round(data[i,1]*100)*100


    for i in range(test_data.shape[0]):
        test_data[i,1] = round(test_data[i,1]*100)*100

    min_percent = min(min(data[:,1]),min(test_data[:,1]))
    max_percent = max(max(data[:,1]),max(test_data[:,1]))

    bins = np.arange(min_percent,max_percent+1,100)

    return data,test_data,bins


def generate_pi(test_data,normalize,theta,beta):
    """
    Generate a policy for the test data set, based on the values of theta and
    beta learned from the training data set.
    Inputs :
        test_data (np.array) : the test data as an array
        normalize (float) : a multiplicative value to normalize how many nearby
                            states are being used to approximate any given state
        theta (dict) : the parameters learned by Q-learning
        beta (dict) : a dictionary that returns the states that are "nearby"
                      a given state (since not every state will be visited)
    Returns :
        pi (dict) : a policy for buying/selling/holding shares, where the keys
                    correspond to the day, and the values correspond to the
                    action for the i^{th} day
    """
    pi = {}
    s = (test_data[0,1],0)

    for i in range(test_data.shape[0]-1):
        flat_s = s[0] + s[1]
        if s[1] == 99:
            values = [normalize*sum(theta[a][beta[flat_s]]) for a in [0,2]]
            if values[1] > values[0]:
                pi[flat_s] = 2
            else:
                pi[flat_s] = 0
        elif s[1] == 0:
            values = [normalize*sum(theta[a][beta[flat_s]]) for a in [0,1]]
            pi[flat_s],_ = max(enumerate(values), key=operator.itemgetter(1))
        else:
            values = [normalize*sum(theta[a][beta[flat_s]]) for a in [0,1,2]]
            pi[flat_s],_ = max(enumerate(values), key=operator.itemgetter(1))

        if i < test_data.shape[0]-1:
            if pi[flat_s] == 1:
                s = (test_data[i+1,1],s[1]+1)
            elif pi[flat_s] == 2:
                s = (test_data[i+1,1],s[1]-1)
            else:
                s = (test_data[i+1,1],s[1])

    return pi



def Qlearn(data,test_data,bins,max_action):
    """
    Run Q-learning with local approximation to learn beta and theta from the
    training data.
    Inputs :
        data (np.array) : the training data as an array
        test_data (np.array) : the test data as an array
        bins (np.array) : the range of daily percent change values possible
        max_action (int) : the number of possible actions to take (e.g. 3 for
                           buy, sell, and hold)
    Returns :
        train_results (list) : the reward accrued on each day of the training set
        test_results (list) : the reward accrued on each day of the test set
        k (int) : the number of iterations through the training data set
        trainplot (pyplot fig) : a plot of the reward accrued on each day of the
                                 training set
        testplot (pyplot fig) : a plot of the reward accrued on each day of the
                                test set
    """
    exp_param = 0
    alpha = 0.05
    nn_bin = 15
    nn_stock = 1
    normalize = 1/((2*nn_bin+1)*(2*nn_stock+1))
    valid_stocks = 100
    gamma = 0.7

    theta = {}
    beta = {}

    train_results = []
    test_results = []

    for b,bin_val in enumerate(bins):
        for st in range(valid_stocks):
            beta[int(bin_val+st)] = [int(bins[min(max(0,i+b),len(bins)-1)]) + min(max(0,j+st),valid_stocks-1) for i in range(-nn_bin,nn_bin+1) for j in range(-nn_stock,nn_stock+1)]

    for i in range(max_action):
        theta[i] = np.zeros(len(bins)*valid_stocks)


    print('starting')
    k = 0
    KeepIterating = True
    while KeepIterating:
        k += 1
        if k%50 == 0:
            print(k, evaluate_2(generate_pi(test_data,normalize,theta,beta),test_data),evaluate_2(generate_pi(data,normalize,theta,beta),data))
            train_results.append(evaluate_2(generate_pi(data,normalize,theta,beta),data))
            test_results.append(evaluate_2(generate_pi(test_data,normalize,theta,beta),test_data))

            if k>100:
                if train_results[-1] == train_results[-2] and train_results[-2] == train_results[-3] and test_results[-1] == test_results[-2] and test_results[-3] == test_results[-2]:
                    KeepIterating = False
                    _,trainplot = evaluate_2(generate_pi(data,normalize,theta,beta),data,return_plot = True)
                    _,testplot = evaluate_2(generate_pi(test_data,normalize,theta,beta),test_data,return_plot = True)
            if k >= 1000:
                KeepIterating = False
                _,trainplot = evaluate_2(generate_pi(data,normalize,theta,beta),data,return_plot = True)
                _,testplot = evaluate_2(generate_pi(test_data,normalize,theta,beta),test_data,return_plot = True)

            if k%100 == 0:
                exp_param += 0.5

        s = (data[0,1],0)
        for i in range(data.shape[0]-1):
            flat_s = s[0]+s[1]
            if s[1] == 0:
                valid_actions = [0,1]
            elif s[1] == 99:
                valid_actions = [0,2]
            else:
                valid_actions = [0,1,2]
            values = [np.exp(exp_param*normalize*sum(theta[a][beta[flat_s]])) for a in valid_actions]
            if np.inf in values:
                at = valid_actions[values.index(np.inf)]
            else:
                at = random.choice(valid_actions, p=[i/sum(values) for i in values])

            if at == 0:
                sp = (data[i+1,1], s[1])
            elif at == 1:
                sp = (data[i+1,1], s[1]+1)
            elif at == 2:
                sp = (data[i+1,1], s[1]-1)
            flat_sp = sp[0]+sp[1]

            r = (data[i+1,0]/1000000)*sp[1]*data[i,0]
            if sp[1] == 0:
                ma = [0,1]
            elif sp[1] == 99:
                ma = [0,2]
            else:
                ma = [0,1,2]

            theta[at][beta[flat_s]] += normalize*(alpha*r + gamma*alpha*(max([normalize*sum(theta[a][beta[flat_sp]]) for a in ma]) - theta[at][beta[flat_s]]))

            s = np.copy(sp)


    return train_results,test_results,k,trainplot,testplot


def main():
    # Use stock values from previous six years as train and test sets.
    train = 'GOOGL_train_old_new.csv'
    test = 'GOOGL_test_old_new.csv'

    # There are three possible actions on each day: buy, sell, and do nothing.
    max_action = 3

    train_data,test_data, bins = preprocess(train,test,max_action)
    len_train = train_data.shape[0]
    len_test = test_data.shape[0]
    train_results,test_results,k,trainplot,testplot = Qlearn(train_data,test_data,bins,max_action)

    # Calculate the reward earned on the training and test sets from acting randomly.
    r1 = 0
    r2 = 0
    for _ in range(100000):
        r1 += evaluate(random_actions(len_train),train_data)
        r2 += evaluate(random_actions(len_test),test_data)
    r1 /= 100000
    r2 /= 100000

    print(train_results[-1],r1,test_results[-1],r2)

    plt.figure(1)
    x = range(0,k,50)
    plt.plot(x,train_results,'b')
    plt.plot(x,test_results,'r')
    plt.plot(x,[r1]*len(x),'c')
    plt.plot(x,[r2]*len(x),'m')
    plt.legend(['Q-learning train','Q-learning test','Random train','Random test'])
    plt.xlabel('number of iterations through dataset')
    plt.ylabel('Net change in wealth')

    plt.figure(2)
    x = range(0,k,50)
    plt.plot(x,test_results,'r')
    plt.plot(x,[r2]*len(x),'m')
    plt.legend(['Q-learning test','Random test'])
    plt.xlabel('number of iterations through dataset')
    plt.ylabel('Net change in wealth')

    plt.figure(3)
    x = range(0,k,50)
    plt.plot(x,train_results,'b')
    plt.plot(x,[r1]*len(x),'c')
    plt.legend(['Q-learning train','Random train'])
    plt.xlabel('number of iterations through dataset')
    plt.ylabel('Net change in wealth')

    plt.figure(4)
    plt.plot(range(0,len(trainplot)),trainplot)
    plt.xlabel('day, starting at Jan 2 2013')
    plt.ylabel('number of shares')

    plt.figure(5)
    plt.plot(range(0,len(testplot)),testplot)
    plt.xlabel('day, starting at Jan 2 2018')
    plt.ylabel('number of shares')


    plt.show()

if __name__ == '__main__':
    main()
