####### Authors: Yicong Lin and Mingxuan Song #######

#==================================================#
# Researchers are free to use the code.
# Please cite the following papers:
# 1. Yicong Lin and Mingxuan Song (2023). Robust bootstrap inference for linear time-varying coefficient models: Some Monte Carlo evidence.
# 2. Marina Friedrich and Yicong Lin (2022). Sieve bootstrap inference for linear time-varying coefficient models. Journal of Econometrics.
######################################################
### Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.ar_model import AutoReg
import math
import warnings
warnings.filterwarnings('ignore')
import scipy

###########################################################
#### Estimation
def K(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0)

def Sn(k, tau, mX, h, times):
    u = (times[None, :] - tau) / h
    K_u = K(u)

    if mX.ndim==2:
        return np.sum((mX[:, None, :] * mX[:, :, None]) * np.reshape(((times[None, :] - tau) ** k) * K_u,
                                                                 newshape=(len(times), 1, 1)), axis=0) / (h)
    elif mX.ndim==1:
        return np.sum((mX[:, None] * mX[:,  None]) * np.reshape(((times[None, :] - tau) ** k) * K_u,
                                                         newshape=(len(times), 1)), axis=0) / (h)
def Tn(k, tau, mX, vY, h, times):
    u = (times[None, :] - tau) / h
    K_u = K(u)

    if mX.ndim==2:
        return np.sum(mX[:, :, None] * np.reshape((times[None, :] - tau) ** k * K_u, newshape=(len(times), 1, 1)) * vY.reshape(
            len(times), 1, 1),axis=0) / (h)
    elif mX.ndim==1:
        return np.sum(mX[:,  None] * np.reshape((times[None, :] - tau) ** k * K_u, newshape=(len(times), 1)) * vY.reshape(len(times), 1 ),axis=0) / (h)

def get_mS(tau, mX, h, times, n_dim):
    mS = np.zeros(shape=(n_dim, n_dim))
    Sn0 = Sn(0, tau, mX, h, times)
    Sn1 = Sn(1, tau, mX, h, times)
    Sn2 = Sn(2, tau, mX, h, times)
    size = Sn0.shape[0]

    mS[:size, :size] = Sn0
    mS[:size, size:] = Sn1
    mS[size:, :size] = Sn1.T
    mS[size:, size:] = Sn2

    return mS

def get_mT(tau, mX, vY, h, times, n_dim):
    mT = np.zeros(shape=(n_dim, 1))
    Tn0 = Tn(0, tau, mX, vY, h, times)
    Tn1 = Tn(1, tau, mX, vY, h, times)
    size = Tn0.shape[0]

    if n_dim == 2:
        mT[:size, 0] = Tn0
        mT[size:, 0] = Tn1
    else:
        mT[:size, 0] = Tn0[:, 0]
        mT[size:, 0] = Tn1[:, 0]

    return mT

def estimator(vY, mX, h, tau, times, n_est):
    betahat = np.zeros(shape=(n_est, len(tau)))
    for i in range(len(tau)):
        mS, mT = get_mS(tau[i], mX, h, times, 2*n_est), get_mT(tau[i], mX, vY, h, times, 2*n_est)
        mul = np.linalg.inv(mS) @ mT

        for j in range(n_est):
            betahat[j, i] = mul[j] if j < n_est else 0

    return betahat

#### Bootstrap setting
def AR(zhat,T):
    maxp = 10 * np.log10(T)
    arm_selection = ar_select_order(zhat, ic='aic', trend='n', maxlag=int(maxp))

    if arm_selection.ar_lags is None:  ######
        armodel = AutoReg(zhat, trend='n', lags=0).fit()
        max_lag = 0  ## to avoid the nonetype error
        epsilonhat = zhat
        epsilontilde = epsilonhat - np.mean(epsilonhat)
    else:
        armodel = arm_selection.model.fit()
        max_lag = max(arm_selection.ar_lags)
        epsilonhat = armodel.resid
        epsilontilde = epsilonhat - np.mean(epsilonhat)

    return epsilontilde, max_lag, armodel


def S_BT(epsilontilde, max_lag, zhat, armodel, mX, betatilde,T):
    epsilonstar = np.random.choice(epsilontilde, T - max_lag + 50)
    if max_lag == 0:
        zstar = epsilonstar[50:]
        zstar_array = zstar
    else:
        max_lag = np.arange(1, max_lag + 1)
        zstar_array = get_Zstar_AR(max_lag, armodel, T, epsilonstar)

    if mX.ndim==1:
        vYstar = (mX * betatilde[0] + zstar_array)
        return vYstar
    elif mX.ndim==2:
        vYstar = (mX @ betatilde + zstar_array).diagonal()
        return vYstar

def SW_BT(epsilontilde, max_lag, zhat, armodel, mX, betatilde,T):
    epsilonstar = epsilontilde * np.random.normal(0, 1, T - max_lag)
    epsilonstar = np.random.choice(epsilonstar, T - max_lag + 50)
    if max_lag == 0:
        zstar = epsilonstar[50:]
        zstar_array = zstar
    else:
        max_lag = np.arange(1, max_lag + 1)
        zstar_array = get_Zstar_AR(max_lag, armodel, T, epsilonstar)

    if mX.ndim==1:
        vYstar = (mX * betatilde[0] + zstar_array)
        return vYstar
    elif mX.ndim==2:
        vYstar = (mX @ betatilde + zstar_array).diagonal()

        return vYstar

def get_Zstar_AR(max_lags, armodel, T, epsilonstar):
    # Initialize the AR process with the known initial values
    zstar = np.zeros(len(max_lags))

    # Add the AR component for each lag value and coefficient
    for i in range(len(max_lags), T):
        ar_component = 0
        for j, lag in enumerate(max_lags):
            lagged_data = zstar[i - lag]
            ar_component += armodel.params[j] * lagged_data

        ar_component += epsilonstar[i + 20 - len(max_lags)]

        zstar = np.append(zstar, ar_component)

    return zstar

def MB_BT(zhat, mX, betatilde,T):
    l = int(1.75 * T ** (1 / 3))

    number_blocks = T - l + 1
    overlapping_blocks = np.zeros(shape=(number_blocks, l, 1))
    for i in range(number_blocks):
        overlapping_blocks[i] = np.array(zhat[i:i + l]).reshape(l, 1)

    random_choice = np.random.choice(np.arange(0, T - l + 1), size=math.ceil(T / l))
    overlapping_blocks_star = overlapping_blocks[random_choice]
    zstar = overlapping_blocks_star.reshape(math.ceil(T / l)*l, 1)[:T]

    if mX.ndim==1:
        # vYstar = np.reshape(mX*betatilde[0],(T,1))+ zstar
        vYstar = (mX * betatilde[0] + zstar).diagonal()
        return vYstar
    elif mX.ndim==2:
        vYstar = (mX @ betatilde + zstar).diagonal()
        return vYstar

#### Simultaneous bands
def get_qtau(alphap, diff, tau):
    qtau = np.zeros(shape=(2, len(tau)))
    for i in range(len(tau)):
        qtau[0, i] = np.quantile(diff[:, i], alphap / 2)

        qtau[1, i] = np.quantile(diff[:, i], (1 - alphap / 2))
    return qtau

def ABS_value(qtau, diff, tau):
    B = 1299
    check = np.sum((qtau[0][:, None] < diff[:, :, None]) & (diff[:, :, None] < qtau[1][:, None]), axis=1)

    return np.abs((np.sum(np.where(check == len(tau), 1, 0)) / B) - 0.95)


def min_alphap(diff, tau):
    B = 1299
    last = ABS_value(get_qtau(1 / B, diff, tau), diff, tau)
    for (index, alphap) in enumerate(np.arange(2, 1299) / 1299):
        qtau = get_qtau(alphap, diff, tau)
        value = ABS_value(qtau, diff, tau)
        if value <= last:
            last = value
            if index == 63:
                return 0.05
        else:
            if index == 0:
                return (index + 1) / B
            if index == 1:
                return (index) / B
            else:
                return (index + 1) / B

#### Bandwidth selection

def omega(x,tau):
    return scipy.stats.norm.pdf(x, loc=tau, scale=np.sqrt(0.025))

def estimation_lmcv(vY,mX,h,tau,times,n_est,lmcv_type):

    T=len(vY)
    betahat = np.zeros(shape=(n_est, len(tau)))

    if lmcv_type==0:
        for i in range(len(tau)):
            new_taut = np.delete(times, i)
            new_mX = np.delete(mX, i, axis=0)
            new_vY = np.delete(vY, i)

            mS, mT = get_mS(tau[i], new_mX, h, new_taut, 2 * n_est), get_mT(tau[i], new_mX, new_vY, h, new_taut,2 * n_est)
            mul = np.linalg.inv(mS) @ mT
            for j in range(n_est):
                betahat[j, i] = mul[j] if j < n_est else 0
    else:
        for i in range(len(tau)):
            deleted_indices = []
            for j in range(lmcv_type+1):
                deleted_indices.append(i - j)
                deleted_indices.append((i + j) % T)
            new_taut=np.delete(times,deleted_indices)
            new_mX=np.delete(mX,deleted_indices,axis=0)
            new_vY=np.delete(vY,deleted_indices)

            mS,mT=get_mS(tau[i],new_mX,h,new_taut,2*n_est),get_mT(tau[i],new_mX,new_vY,h,new_taut,2*n_est)
            mul=np.linalg.inv(mS)@mT
            for j in range(n_est):
                betahat[j, i] = mul[j] if j < n_est else 0

    return betahat

def LMCV(betahat_lmcv,mX,vY,one_tau):
    T = len(vY)

    taut = np.arange(1/T,(T+1)/T,1/T)
    if mX.ndim==1:
        aa=(vY-(mX*betahat_lmcv))**2
        b = omega(taut, one_tau)

        return np.sum(aa * b) / T
    elif mX.ndim==2:
        aa=(vY-(mX@betahat_lmcv).diagonal())**2
        b = omega(taut, one_tau)

        return np.sum(aa * b) / T

def get_optimalh(vY,mX,lmcv_type,n_est):
    T = len(vY)

    taut = np.arange(1/T,(T+1)/T,1/T)
    optimal_h_tau=[]
    vh=np.arange(0.06,0.2,0.005)

    betasss=np.zeros(shape=(len(vh),n_est,T))
    for (index,h) in enumerate(vh):
        betasss[index]=estimation_lmcv(vY,mX,h,taut,taut,n_est,lmcv_type)

    for one_tau in taut:
        contain=[]

        for (index,h) in enumerate(vh):
            contain.append(LMCV(betasss[index],mX,vY,one_tau))
        optimal_h_tau.append(np.argmin(np.array(contain)))

    return vh[min(optimal_h_tau)]

def BW_sele(vY,mX,n_est):
    T = len(vY)
    h = []
    taut = np.arange(1 / T, (T + 1) / T, 1 / T)
    for lmcv_type in [0, 2, 4, 6]:
        h.append(get_optimalh(vY,mX, lmcv_type, n_est))
    AVG = np.mean(h)
    h.append(AVG)

    return h

#### empirical
def empirical(vY,mX,h,n_est,type):

    htilde=2*(h**(5/9))
    T=len(vY)
    taut=np.arange(1/T,(T+1)/T,1/T)
    B=1299

    betatilde = estimator(vY, mX, htilde, taut, taut,n_est)
    betahat=estimator(vY, mX, h, taut, taut,n_est)

    #### for any potential subset
    # G1=___
    # betatilde_G1 = estimator(vY, mX, htilde,G1,taut,n_est)
    # betahat_G1 = estimator(vY, mX, h,G1,taut,n_est)
    if n_est==1:
        zhat=vY-(mX*betatilde[0])
    else:
        zhat=vY-(mX@betatilde).diagonal()
    betahat_star=np.zeros(shape=(B,n_est,T))
    # betahat_star_G1=np.zeros(shape=(B,5,len(G1)))

    ### For S & SW bootstrap
    if type == 1:
        epsilontilde,max_lag,armodel= AR(zhat,T)
        for i in range(B):
            vYstar=S_BT(epsilontilde,max_lag,zhat,armodel,mX,betatilde,T)
            betahat_star[i]=estimator(vYstar,mX,h,taut,taut,n_est)

    elif type == 2:
        epsilontilde, max_lag, armodel = AR(zhat, T)
        for i in range(B):
            vYstar = SW_BT(epsilontilde, max_lag, zhat, armodel, mX, betatilde, T)
            betahat_star[i] = estimator(vYstar, mX, h, taut, taut, n_est)

    # ### For MB bootstrap
    elif type == 3:
        for i in range(B):
            vYstar=MB_BT(zhat,mX,betatilde,T)
            betahat_star[i]=estimator(vYstar,mX,h,taut,taut,n_est)

    diff_beta = np.zeros(shape=(n_est,B,T))

    for i in range(B):
        diff = betahat_star[i] - betatilde
        for j in range(n_est):
            diff_beta[j][i] = diff[j]

    #### Simultaneous bands
    optimal_alphap = [min_alphap(diff_beta[i], taut) for i in range(n_est)]

    S_LB_beta = [betahat[i] - get_qtau(optimal_alphap[i], diff_beta[i], taut)[1] for i in range(n_est)]
    S_UB_beta = [betahat[i] - get_qtau(optimal_alphap[i], diff_beta[i], taut)[0] for i in range(n_est)]

    P_LB_beta=[betahat[i] - get_qtau(0.05, diff_beta[i], taut)[1] for i in range(n_est)]
    P_UB_beta=[betahat[i] - get_qtau(0.05, diff_beta[i], taut)[0] for i in range(n_est)]
    return S_LB_beta,S_UB_beta,P_LB_beta,P_UB_beta,betahat

####
#### import your data here

# mX=_____
# vY=_____

#### choose h by bandwidth seletion, note: here and later implementation, pure values of mX and vY are needed, so sometimes variables needs to be
#### modified such as "vY.values", and n_est represents the number of the explanatory variables

# vh_BW = BW_sele(vY,mX,n_est)
# print('Bandwidth selected by LMCV0, LMCV2, LMCV4, LMCV6, AVG')
# print(vh_BW)

# vh_BW_AVG = vh_BW[4]  #### defautly select h_AVG as the preference, but one can specify by empirical need
# print('Bandwidth selected by AVG')
# print(vh_BW_AVG)

# S_LB_beta,S_UB_beta,P_LB_beta,P_UB_beta,betahat = empirical(vY, mX, h_preferred, n_est, Bootstrap_type)
##### where S_LB_beta and S_UB_beta are simultaneous bands, P_LB_beta and P_UB_beta are pointwise intervals, betahat are the estimators.
#### The parameter "Bootstrap_type" is for the choice of different bootstrap method:
####     1 is for Sieve BT, 2 is for Sieve Wild BT, 3 is for Moving block BT.

#### The Bootstrap bands are constructed after 1299 times of bootstrap.

# plt.plot(S_LB_beta[])
# plt.plot(S_UB_beta[])
# plt.plot(betahat[])
# plt.show()
