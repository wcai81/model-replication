import pandas as pd
import numpy as np
import math
from sklearn.linear_model import lasso_path, lars_path
import os
from numpy.linalg import eigh
import statsmodels.api as sm

parent_dir ='C:/Users/weixi/PycharmProjects/FFORMA Application/characteristics data Markus/'

#Author: Wenjing Cai
#Object of the code: This is the code for pruning the asset pricing decision trees using elastic net shrinkage to all final and intermediate nodes of asset pricing trees.
#This code can refer to the paper: Bryzgalova, S., Pelger, M., & Zhu, J. (2019). Forest through the trees: Building cross-sections of stock returns. Available at SSRN 3493458.


#The function lasso_valid_full is to do 3 folds cross validation and tune the hyperparameter by maximize the out-of-sample Sharpe ratio

def lasso_valid_full(ports, lambda0, lambda2, main_dir,subdir, adj_w, n_train_valid = 360, cvN = 3, kmin=0, kmax=20):

    ports_test = ports.iloc[(n_train_valid):(len(ports.index)),:]
    n_valid = n_train_valid / cvN

    n_train = n_train_valid - n_valid

    # 3 folds cross validation
    lambda0_opt_cv=[]
    lambda2_opt_cv=[]
    k_opt_cv=[]
    SR_opt_cv=[]
    for i in range(1, (cvN+1)):
        train_idx = list(range(int((i - 1) * n_valid + 1))) + list(range(int(i * n_valid), int(n_train_valid + 1)))
        ports_train = ports.iloc[train_idx]
        ports_valid = ports.iloc[int((i - 1)*n_valid+1):int(i * n_valid), :]
        lambda0_opt,lambda2_opt,k_opt,SR_opt=lasso_cv_helper(ports_train, ports_valid, lambda0, lambda2, adj_w, kmin,kmax)
        lambda0_opt_cv.append(lambda0_opt)
        lambda2_opt_cv.append(lambda2_opt)
        k_opt_cv.append(k_opt)
        SR_opt_cv.append(SR_opt)

    # Method: choose optimal tunning parameter by choose max SR across different folds
    lambda0_final=lambda0_opt_cv[np.argmax(SR_opt)]
    lambda2_final=lambda2_opt_cv[np.argmax(SR_opt)]
    k_final=k_opt_cv[np.argmax(SR_opt)]
    print(f'lambda0_final: {lambda0_final}')
    print(f'lambda2_final: {lambda2_final}')
    print(f'k_final: {k_final}')
    # After pin-down the parameter, do another fit on the whole train+valid time period
    ports_train = ports.iloc[0:n_train_valid,:]
    final_results=lasso_final_helper(ports_train, ports_test, lambda0_final, lambda2_final, adj_w,k_final)
    final_results.to_csv(main_dir+'/'+subdir+'/'+subdir+'/'+'results_tc_s_r'+ '.csv')
    return final_results
    # Call LARS to calculate the whole path for EN regularized regression

# The function lasso is to define the elastic net model by using lars algorithm for cross validation
def lasso(X, y, lambda2, kmin,kmax):
    n = X.shape[0]
    p = X.shape[1]
    yy = np.concatenate((y, np.zeros((p,), dtype=int)), axis=None)
    XX = np.concatenate((X, np.diag(np.repeat(math.sqrt(lambda2), p))),
                        axis=0)

    alpha, active, beta, n_iter = lars_path(XX, yy, method='lasso', max_iter=12056, Gram='auto', return_n_iter=True)

    K=np.count_nonzero(beta, axis=0)

    print(f'shape of original K: {K.shape}')

    subset = np.where((K>=kmin)&(K<=kmax))

    print(f'shape of original beta: {beta.shape}')
    print(f'subset is: {subset}')

    return [np.squeeze(beta[:,subset], axis=1), K[subset]]

# The function lasso_cv_helper is to train the elastic net model defined in the function lasso for cross validation
def lasso_cv_helper(ports_train, ports_valid, lambda0, lambda2, adj_w, kmin, kmax):
    # Converting the optimization into a regression problem
    mu = ports_train.mean()
    sigma = ports_train.cov()

    mu_bar = mu.mean()
    gamma = np.min(ports_train.shape)
    eigenValues, eigenVectors = eigh(sigma)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    sum=0
    for j in range(0,len(eigenValues)):
        if eigenValues[j] > 10 ** (-10):
            sum += 1
    gamma = min(gamma, sum)
    D = eigenValues[0:gamma]
    V = eigenVectors[:, 0:gamma]
    a=np.zeros((gamma,gamma),float)
    np.fill_diagonal(a,np.sqrt(D))
    sigma_tilde=np.dot(np.dot(V,a),np.transpose(V))
    b = np.zeros((gamma, gamma), float)
    np.fill_diagonal(b, 1/np.sqrt(D))
    c=np.transpose(np.tile(np.array(mu), len(lambda0)).reshape(len(lambda0),len(mu)))
    d=np.transpose((np.repeat(np.array(lambda0),len(mu))*mu_bar).reshape(len(lambda0),len(mu)))
    mu_tilde = np.dot(np.dot(np.dot(V,b),np.transpose(V)),(c+d))
    e = np.zeros((gamma, gamma), float)
    np.fill_diagonal(e, 1 / D)
    w_tilde = np.dot(np.dot(np.dot(V,e),np.transpose(V)),(c+d))

    # Perform EN regression with CPU parallel computing
    cv_set=pd.DataFrame()
    for i in range(0,len(lambda0)):
        for j in range(0,len(lambda2)):
            lasso_results = lasso(sigma_tilde, mu_tilde[:, i], lambda2[j], kmin, kmax)
            print(f'shape of sigma_tilde is {sigma_tilde.shape}')
            print(f'shape of mu_tilde is {mu_tilde[:, i].shape}')


            train_SR=np.zeros(lasso_results[0].shape[1])
            valid_SR=np.zeros(lasso_results[0].shape[1])

            betas = np.zeros((lasso_results[0].shape[1], len(mu)))
            portsN = lasso_results[1]

            for r in range(0,lasso_results[0].shape[1]):
              b = (lasso_results[0].T)[r, : ]

              print(f'shape of b: {b.shape}')
              print(f'shape of non zero beta: {lasso_results[0].shape}')
              print(f'shape of adj_w: {adj_w.shape}')
              print(f'shape of filtered K: {lasso_results[1].shape}')

              #Adjust the weight by depth
              b = b * adj_w
              b = b / abs(np.sum(b))

              sdf_train = np.dot(ports_train.values,(b / adj_w))
              sdf_valid = np.dot(ports_valid.values, (b / adj_w))
              train_SR[r] = np.mean(sdf_train) / np.std(sdf_train)
              valid_SR[r] = np.mean(sdf_valid) / np.std(sdf_valid)
              betas[r,: ] = b

            results = pd.DataFrame(data=betas, columns=ports_train.columns)
            results['train_SR'] = train_SR
            results['portsN'] = portsN
            results['valid_SR'] = valid_SR
            results['lambda0']=lambda0[i]
            results['lambda2']=lambda2[j]
            cv_set=cv_set.append(results)
    #Choose the tunning parameter combination with the highest validation sharp ratio
    lambda0_opt=cv_set['lambda0'].iloc[np.argmax(cv_set['valid_SR'])]
    lambda2_opt=cv_set['lambda2'].iloc[np.argmax(cv_set['valid_SR'])]
    k_opt=cv_set['portsN'].iloc[np.argmax(cv_set['valid_SR'])]
    SR_opt=max(cv_set['valid_SR'])
    return lambda0_opt,lambda2_opt,k_opt,SR_opt

# The function lasso_final is to define the elastic net model by using lars algorithm for final training on the whole model after hyperparameter chosen
def lasso_final(X, y, lambda2, k):
    n = X.shape[0]
    p = X.shape[1]
    yy = np.concatenate((y, np.zeros((p,), dtype=int)), axis=None)
    XX = np.concatenate((X, np.diag(np.repeat(math.sqrt(lambda2), p))),axis=0)

    alpha, beta, dual_gaps = lasso_path(XX,yy)  # beta shape: (n_features, n_alphas)

    K=np.count_nonzero(beta, axis=0)

    print(f'shape of original K: {K.shape}')

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = np.where(np.abs(array - value)==(np.abs(array - value)).min())

        return idx

    if k in K:
      subset = np.where(K==k)
    else:
       subset = find_nearest(K, k)
    print(f'shape of original beta: {beta.shape}')
    print(f'subset is: {subset}')
    return [np.squeeze(beta[:,subset], axis=1), K[subset]]


# The function lasso_final_helper is to train the elastic net model defined in the function lasso_final for final training on the whole model after hyperparameter chosen
def lasso_final_helper(ports_train,ports_test,lambda0,lambda2,adj_w,k):
    # Converting the optimization into a regression problem
    mu = ports_train.mean()
    sigma = ports_train.cov()


    mu_bar = mu.mean()
    gamma = np.min(ports_train.shape)
    eigenValues, eigenVectors = eigh(sigma)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    sum=0
    for j in range(0,len(eigenValues)):
        if eigenValues[j] > 10 ** (-10):
            sum += 1
    gamma = min(gamma, sum)
    D = eigenValues[0:gamma]
    V = eigenVectors[:, 0:gamma]
    a=np.zeros((gamma,gamma),float)
    np.fill_diagonal(a,np.sqrt(D))
    sigma_tilde=np.dot(np.dot(V,a),np.transpose(V))
    b = np.zeros((gamma, gamma), float)
    np.fill_diagonal(b, 1/np.sqrt(D))
    c=np.array(mu)
    d=np.repeat(np.array(lambda0),len(mu))*mu_bar
    mu_tilde = np.dot(np.dot(np.dot(V,b),np.transpose(V)),(c+d))
    e = np.zeros((gamma, gamma), float)
    np.fill_diagonal(e, 1 / D)
    w_tilde = np.dot(np.dot(np.dot(V,e),np.transpose(V)),(c+d))

    # Perform EN regression with CPU parallel computing
    lasso_results = lasso_final(sigma_tilde, mu_tilde, lambda2, k)

    portsN = lasso_results[1]

    b = lasso_results[0][:,0]
    b = b * adj_w
    b = b / abs(np.sum(b))

    sdf_train = np.dot(ports_train.values,(b / adj_w))
    sdf_test = np.dot(ports_test.values,(b / adj_w))
    train_SR = np.mean(sdf_train) / np.std(sdf_train)
    test_SR = np.mean(sdf_test) / np.std(sdf_test)
    betas = b
    results = pd.DataFrame(data=betas).T
    results.columns = ports_train.columns
    results['train_SR'] = train_SR
    results['test_SR'] = test_SR
    results['portsN'] = portsN
    return results


if __name__ == "__main__":
    # Step 1 get the trees based on the any combination of 3 features chosed from the list
    feats_list = ['LME', 'BEME', 'r12_2', 'OP', 'Investment', 'ST_Rev','LT_Rev', 'AC', 'IdioVol', "LTurnover"]

    excludes = []

    n_train_valid = 360
    cvN = 3

    for f1 in range(0, 1):
         for f2 in range((f1 + 1), 9):
             for f3 in range((f2 + 1), 10):

               chars = [feats_list[f1], feats_list[f2], feats_list[f3]]

               main_dir = os.path.join(parent_dir, 'lasso_full_valid')

               subdir = str(chars[0]) + '_' + str(chars[1]) + '_' + str(chars[2])
               os.mkdir(main_dir+'/'+subdir+'/'+subdir)
               print(subdir)

               ports = pd.read_csv(os.path.join(parent_dir, subdir+'/'+subdir, "level_all_excess_combined_filtered.csv"))

               ports = ports.iloc[:, 1:]
    # only minus 6 since when python read the dataset, the columns don't have X
               depths = ports.columns.str.len() - 6

               coln = ports.columns
               coln_1 = {}
               for i in range(0, len(coln)):
                  coln_1[ports.columns[i]] = coln[i][0:(len(coln[i][5:]) - 1)] + '.' +coln[i][5:]
               ports.rename(columns=coln_1, inplace=True)

               adj_w = 1 / np.sqrt(np.power(2, depths))

               adj_ports = ports
               for i in range(0, len(adj_w)):
                   adj_ports.iloc[:, i] = ports.iloc[:, i] * adj_w[i]


               lambda0 = np.arange(0, 0.2, 0.05)

               lambda2 = np.power(0.1, np.arange(5, 8, 3))

               # Step 2 pruning the trees using elastic net model
               lasso_valid_full(adj_ports, lambda0, lambda2, main_dir, subdir, adj_w, n_train_valid, cvN)
