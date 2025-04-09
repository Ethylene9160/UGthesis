import numpy as np
import matplotlib.pyplot as plt
from pylab import *

def activation_thresh(x, sigma=0.0, c_thresh=None):
    '''
    Apply a threshold-based activation function to an input array.

    This function modifies the input array based on a threshold value. Elements in the array whose absolute value is below the threshold are set to zero. The function retains the sign of the original elements that are above the threshold.

    Parameters:
        x (numpy.ndarray): The input array to which the activation function is applied. This should be a numpy array.
        sigma (float, optional): Currently unused parameter, reserved for future extensions or modifications. Defaults to 0.0.
        c_thresh (float, optional): The cutoff threshold value below which values in 'x' are set to zero. If None, the threshold is automatically calculated as 2 divided by the square root of the size of 'x'. Defaults to None.

    Returns:
        numpy.ndarray: The array resulting after applying the threshold-based activation. Elements below the threshold are zeroed, while elements above the threshold retain their original value and sign.

    '''
    
    # todo: Why the threshold is set to 2/sqrt(N)?
    if c_thresh is None:
        N = x.shape[0]
        c_thresh = 2.0 / N**0.5
        
    xn = np.abs(x)
    
    a = (x ) / (np.abs(x) + 1e-12)
    a[xn < c_thresh] = 0
    
    return a

def crvec(N, D=1):
    rphase = 2*np.pi * np.random.rand(D, N)
    return np.cos(rphase) + 1.0j * np.sin(rphase)

def crevc_dense(N, D, k):
    rphase = 2*np.pi * np.random.rand(D, N)
    return np.cos(rphase) + 1.0j * np.sin(rphase)

def phase2spikes(cv, freq=5.0):
    '''
    
    Args:
        cv: complex vector
        freq: frequency

    Returns: spike times. i.e. if the phase is pi, and the frequency is 5Hz, the spike time is 0.1s

    '''
    st = np.angle(cv) / (2*pi*freq)
    return st

def STDP_f(delta_t, W_curr, Ap=0.0004, An=0.0004, tau_p=200, tau_n=200):
    """
    改进的 STDP 更新规则，返回复数权重的变化。
    Args:
        delta_t: 时间差矩阵 (valid_i, valid_j)
        W_curr: 当前权重矩阵 (valid_i, valid_j)
        Ap: 突触增强幅度
        An: 突触削弱幅度
        tau_p: 增强的时间常数
        tau_n: 削弱的时间常数
    Returns:
        突触权重更新矩阵（复数）
    """
    # 计算突触幅值的变化
    delta_W_mag = np.where(delta_t > 0, 
                           Ap * np.exp(-delta_t / tau_p),  # LTP
                           An * np.exp(delta_t / tau_n))  # LTD
    # delta_W_mag = np.where(delta_t < 0, 
    #                        1.01,  # LTP
    #                        0.99)  # LTD
    return delta_W_mag

def stdp_learn(cvs, epochs=1, Ap=1.05, An=1.05, tau_p=1.2, tau_n = 1.2, freq = 5.0):
    """
    STDP 学习函数，修复索引超出范围的问题。
    Args:
        cvs: complex vectors (N_vec, N_out)
        epochs: 学习的迭代次数
        Ap: 突触增强系数
        An: 突触削弱系数
        tau: 时间常数
    Returns:
        复数权重矩阵 (N_vec, N_vec)
    """
    N_vec, N_out = cvs.shape  # 输入矩阵的大小
    W = np.zeros((N_vec, N_vec), dtype=np.complex64)  # 权重矩阵初始化

    # 提前计算相位存储，归一化到 5Hz
    angle_store = np.angle(cvs)  # 输入矩阵的相位
    phase_store = angle_store / (2 * np.pi * freq)  # 输入矩阵的相位
    valid_mask = np.abs(cvs) > 0.05  # 稀疏矩阵的非零掩码

    for epoch in range(epochs):
        print(f'{epoch}th training epoch')

        # 遍历每一列对
        for i in range(N_out):
            for pre in range(N_vec):
                for post in range(N_vec):
                    if valid_mask[pre, i] and valid_mask[post, i]:
                        delta_angle = angle_store[pre, i] - angle_store[post, i]
                        delta_t = (phase_store[pre, i] - phase_store[post, i])
                        # print(f'delta_t: {delta_t}, delta_angle: {delta_angle}, {delta_t*2*pi*5} =  {delta_angle}')
                        # print('abs t: ', STDP_f(np.abs(delta_t), W[pre, post], Ap=Ap, An=An, tau_p=tau, tau_n=tau)*np.exp(1j*(delta_angle)))
                        # print('abs theta: ', STDP_f((delta_angle), W[pre, post], Ap=Ap, An=An, tau_p=tau, tau_n=tau)*np.exp(1j*np.abs(delta_angle)))
                        W[pre, post] += STDP_f(delta_t, W[pre, post], Ap=Ap, An=An, tau_p=tau_p, tau_n=tau_n)*np.exp(1j*(delta_angle))
            if i % (N_out/10) == 0:
                print(f'{i/N_out*100}%th column processed')
                

    # 确保对角线无自连接
    # np.fill_diagonal(W, 0)
    return W




def cviz_im(cvec):
    ss = int(len(cvec)**0.5)
    
    ss_idx = ss**2
    
    im_cvec = np.zeros((ss, ss,3))
#     im_cvec[:,:,3]=1
    c=0
    for i in range(ss):
        for j in range(ss):
            if np.abs(cvec[c]) > 0.05:
                im_cvec[i,j,:] = matplotlib.colors.hsv_to_rgb([(np.angle(cvec[c])/2/pi + 1) % 1, 1, 1])
                
            c+=1
                
    return im_cvec

def save_complex_matrix(filename, matrix):
    """
    保存复数矩阵到文件（.npz 格式）。
    Args:
        filename (str): 保存的文件名，例如 'pvecs.npz'
        matrix (ndarray): 要保存的复数矩阵
    """
    # 将复数分解为实部和虚部
    np.savez(filename, real=matrix.real, imag=matrix.imag)
    print(f"Matrix saved to {filename}")
    
def load_complex_matrix(filename):
    """
    从文件中读取复数矩阵（.npz 格式）。
    Args:
        filename (str): 保存的文件名，例如 'pvecs.npz'
    Returns:
        ndarray: 还原的复数矩阵
    """
    # 从文件中加载数据
    data = np.load(filename)
    real = data['real']
    imag = data['imag']
    matrix = real + 1j * imag
    print(f"Matrix loaded from {filename}")
    return matrix
