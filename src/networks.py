from neurons import izh_neuron, mizh_neuron, hh_neuron
from neurons import *
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from Data import ONE, ZERO,TWO,SIX,INPUT_ZERO,INPUT_TWO,INPUT_ONE,INPUT_SIX

from matplotlib import rc
from matplotlib import rcParams
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rcParams['font.family'] = 'Times New Roman'
class NEURON_TYPE(Enum):
    IZH = 1
    MIZH = 2
    HH = 3
    LIF = 4

class network:
    def __init__(self, size=64, iter=10, eta=1.0, dt=1.0):
        self.iter = iter
        self.size = size
        self.eta = eta
        self.n = size ** 2
        self.W = np.zeros((size ** 2, size ** 2))
        self.dt = dt
        self.I = np.eye(self.size ** 2) # unit matrix

    # using hebbo method, train the basic network.
    # Formular:
    def train(self, X):
        for x in X:  # (-1,64*64)
            x = np.reshape(x, (self.n, 1))
            xT = np.reshape(x, (1, self.n))
            self.W += self.eta * x * xT
            # self.W[np.diag_indices_from(self.W)] = 0
            self.W -= self.I

    def predict(self, x, model_type:NEURON_TYPE = NEURON_TYPE.MIZH):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x = x.reshape((self.n,)).astype(float)  # Simplify reshaping and type conversion
        unit = mA
        # 建立self.n个mizh_neuron：
        spike_times = []
        spike_indices = []
        neuron_nets = None
        if model_type == NEURON_TYPE.MIZH:
            neuron_nets = [mizh_neuron() for _ in range(self.n)]
        elif model_type == NEURON_TYPE.IZH:
            neuron_nets = [izh_neuron(max_current=1, time_factor=0.18) for _ in range(self.n)]
        elif model_type == NEURON_TYPE.HH:
            unit = 0.32 * mA
            neuron_nets = [hh_neuron(refractory_time = 0.5) for _ in range(self.n)]
        elif model_type == NEURON_TYPE.LIF:
            neuron_nets = [lif_neuron() for _ in range(self.n)]
        # for i in range(self.n):
        #     neuron_nets[i].v = 30 if x[i] == 1 else -65

        # These copies are now moved closer to where they are first modified
        energy = []
        monitor_index = 0

        for i in range(self.n):
            if x[i] == 1:
                # neuron_nets[i].v = 30
                # neuron_nets[i].reset()
                neuron_nets[i].spike_current = 1.0
                neuron_nets[i].last_spike_time = neuron_nets[i].current_time
        # 创建全是-1的一维数组
        x_new = np.array([-1 for _ in range(self.n)])
        i_new = x.copy()*unit

        monitor_iext = [np.dot(self.W[monitor_index], x.copy()*unit)]
        monitor_v = [neuron_nets[monitor_index].get_v()]
        monitor_i = [neuron_nets[monitor_index].get_spike_current()]
        for _ in range(self.iter):
            i_new = [n.get_spike_current() * unit for n in neuron_nets]
            for i in range(self.n):
                iext = np.dot(self.W[i], i_new)
                v = neuron_nets[i].step(iext, self.dt)
                if neuron_nets[i].reset():
                    x_new[i] = 1
                    spike_indices.append(i)
                    spike_times.append(_*self.dt)
                else:
                    x_new[i] = x_new[i]
                # x_new[i] = 1 if neuron_nets[i].reset() else x_new[i]# else -1
            x = x_new.copy()
            energy.append(self.cal_energy(x))
            monitor_iext.append(np.dot(self.W[monitor_index], i_new))
            monitor_v.append(neuron_nets[monitor_index].get_v())
            monitor_i.append(neuron_nets[monitor_index].get_spike_current())
        x = x_new.copy()
        return (x.reshape((self.size, self.size)),
                energy,
                monitor_iext,
                monitor_v,
                monitor_i,
                spike_indices,
                spike_times)
    # an implimention of Lypn.
    def cal_energy(self, x):
        # n = self.size ** 2
        energy = np.sum(self.W.dot(x) * x)
        return -0.5 * energy

if __name__ == "__main__":

    # # show images in the training sets
    # plt.figure(figsize=(5.5, 6))
    # plt.suptitle('Stored Patterns', fontsize=16)
    #
    # plt.subplot(2, 2, 1)
    # plt.imshow(ZERO, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 2)
    # plt.imshow(ONE, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(TWO, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 4)
    # plt.imshow(SIX, cmap='gray')
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.savefig('stored_patterns.eps', format='eps')
    # plt.show()
    #
    # # Input images
    # plt.figure(figsize=(5.5, 6))
    # plt.suptitle('Input Patterns', fontsize=16)
    # plt.subplot(2, 2, 1)
    # plt.imshow(INPUT_ZERO, cmap='gray')
    # plt.axis('off')
    # plt.subplot(2, 2, 2)
    # plt.imshow(INPUT_ONE, cmap='gray')
    # plt.axis('off')
    # plt.subplot(2, 2, 3)
    # plt.imshow(INPUT_TWO, cmap='gray')
    # plt.axis('off')
    # plt.subplot(2, 2, 4)
    # plt.imshow(INPUT_SIX, cmap='gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('input_patterns.eps', format='eps')
    # plt.show()

    size = 9
    # define the label
    label = np.array([ZERO,ONE,TWO,SIX])
    # label = np.array([SIX,SIX])
    # print(label)

    input = INPUT_SIX

    # parameters:
    # MIZH: iter = 250, dt = 0.04, eta = 1.500
    # IZH: iter = 250, dt = 0.04, eta = 1.000
    # HH: iter = 5000, dt = 0.002, eta = 2.1
    # LIF: iter = 250, dt = 0.04, eta = 1.000
    iter = 5000
    dt = 0.002
    eta = 2.1

    model = network(size=size, iter=iter, eta=eta, dt = dt)
    model.train(label)

    model_type = NEURON_TYPE.HH
    # get the predict results
    y0,e0,_,_,_, spike_indices_0, spike_times_0 = model.predict(INPUT_ZERO, model_type)
    y1,e1,_,_,_, spike_indices_1, spike_times_1 = model.predict(INPUT_ONE, model_type)
    y2,e2,_,_,_, spike_indices_2, spike_times_2 = model.predict(INPUT_TWO, model_type)
    y6,e6,monitor_iext,monitor_v, monitor_i, spike_indices_6, spike_times_6 = model.predict(INPUT_SIX, model_type)

    # show the result graphes
    plt.figure(figsize=(8, 4))

    # Input images
    plt.subplot(2, 4, 1)
    plt.imshow(INPUT_ZERO, cmap='gray')
    plt.axis('off')
    plt.title("Input Unordered Zero")

    plt.subplot(2, 4, 2)
    plt.imshow(INPUT_ONE, cmap='gray')
    plt.axis('off')
    plt.title("Input Unordered One")

    plt.subplot(2, 4, 3)
    plt.imshow(INPUT_TWO, cmap='gray')
    plt.axis('off')
    plt.title("Input Unordered Two")

    plt.subplot(2, 4, 4)
    plt.imshow(INPUT_SIX, cmap='gray')
    plt.axis('off')
    plt.title("Input Unordered Six")

    # Output images
    plt.subplot(2, 4, 5)
    plt.imshow(y0, cmap='gray')
    plt.axis('off')
    plt.title("Reconstruct with Input\nUnordered Zero")

    plt.subplot(2, 4, 6)
    plt.imshow(y1, cmap='gray')
    plt.axis('off')
    plt.title("Reconstruct with Input\nUnordered One")

    plt.subplot(2, 4, 7)
    plt.imshow(y2, cmap='gray')
    plt.axis('off')
    plt.title("Reconstruct with Input\nUnordered Two")

    plt.subplot(2, 4, 8)
    plt.imshow(y6, cmap='gray')
    plt.axis('off')
    plt.title("Reconstruct with Input\nUnordered Six")

    plt.tight_layout()
    plt.savefig('reconstruct.eps', format='eps')
    plt.show()

    plt.figure()
    x_values = np.arange(0, len(monitor_i)) * dt
    print(len(x_values))
    print(len(monitor_i))
    plt.subplot(3,1,1)
    plt.plot(x_values, monitor_iext, color='black')
    plt.xlabel('Time/s')
    plt.title(r'$I_{ext0}(t)$')
    plt.subplot(3,1,2)
    plt.plot(x_values, monitor_v, color='black')
    plt.xlabel('Time/s')
    plt.title('$V_0(t)$')
    plt.subplot(3,1,3)
    plt.plot(x_values, monitor_i, color='black')
    plt.xlabel('Time/s')
    plt.title(r'$I_{0}(t)$')
    plt.tight_layout()
    # plt.savefig(f'neuron_0_{model_type.name}.eps', format='eps')
    plt.show()

    plt.figure()
    # plt.subplot(2,2,1)
    # plt.scatter(spike_indices_0, spike_times_0)
    # plt.title('spike times of zero')
    # plt.xlabel('time')
    # plt.ylabel('neuron index')
    # plt.subplot(2,2,2)
    # plt.scatter(spike_indices_1, spike_times_1)
    # plt.title('spike times of one')
    # plt.xlabel('time')
    # plt.ylabel('neuron index')
    # plt.subplot(2,2,3)
    # plt.scatter(spike_indices_2, spike_times_2)
    # plt.title('spike times of two')
    # plt.xlabel('time')
    # plt.ylabel('neuron index')
    # plt.subplot(2,2,4)
    # scatter the spike times of six
    # size should be small..
    plt.scatter(spike_times_6, spike_indices_6, s=1, color='black')
    plt.title('Spike Times of Neurons')
    plt.xlabel('Time/s')
    plt.ylabel('Neuron Index')
    # plt.savefig(f'spike_times_of_{model_type.name}.eps', format='eps')
    plt.show()