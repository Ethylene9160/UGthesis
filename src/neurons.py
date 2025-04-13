import numpy as np
from matplotlib import pyplot as plt

mA = 1.0
uA = 1e-3
nA = 1e-6

def sym_sigmoid(x, time_delay=10):
    expn = np.exp(-10*(x-time_delay))
    return 2.2*expn / (1 + expn)-1.1

class neuron:
    def __init__(self, a, b, c, d, refractory_time = 0.0, max_current = 1.0):
        self.a1 = 0.04
        self.a2 = 5.0
        self.a3 = 140.0
        self.a4 = 1.0
        self.a5 = 1.0

        self.a = a
        self.b = b
        self.v = c
        self.c = c
        self.d = d
        self.u = self.b * self.v

        self.on_refractory = False
        self.refractory_time = refractory_time
        self.current_refractory_time = 0.0
        self.current_time = 0.0
        self.last_spike_time = 0.0
        self.time_factor = 0.18
        self.spike_current = -1.0
        self.max_current = max_current
        self.customized_name = ' '

    def set_refractory(self, time):
        self.refractory_time = time

    def get_v(self):
        return self.v

    def reset(self):
        if self.v >= 30:
            self.on_refractory = True
            self.current_refractory_time = self.refractory_time
            self.spike_current = self.max_current
            self.last_spike_time = self.current_time
            self.v = self.c
            self.u += self.d
            return True
        return False

    def threshold(self):
        if self.v > 35.0:
            self.v = 35.0
        elif self.v < -75.0:
            self.v = -75.0

    def step(self, I, dt) -> float:
        self.current_time += dt
        if self.on_refractory:
            self.current_refractory_time -= dt
            I = 0.0
            if self.current_refractory_time <= 0:
                self.on_refractory = False
        return I

    def _generate_current(self, dt) -> float:
        if self.spike_current == -1.0:
            return -1.0
        if self.current_time - self.last_spike_time > 0.0 and \
            (self.current_time - self.last_spike_time) < 20:
            self.spike_current -= dt*self.time_factor*self.spike_current
        else:
            self.spike_current = -1.0
        return self.spike_current

    def get_spike_current(self):
        return self.spike_current
    
    def spike(self):
        self.v = 30
        return


class izh_neuron(neuron):
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=2.0, 
                 refractory_time = 0.0,
                 time_factor = 1.0,
                 max_current = 1.0):
        super().__init__(a=a, b=b, c=c, d=d, refractory_time = refractory_time)
        self.spike_current = -1
        self.time_factor = time_factor
        self.max_current = max_current
    def step(self, I, dt=0.1):
        # self.current_time += dt
        # if self.on_refractory:
        #     self.current_refractory_time -= dt
        #     I = 0.0
        #     if self.current_refractory_time <= 0:
        #         self.on_refractory = False
        I = super().step(I = I, dt = dt)
        '''
        根据Izh博士的论文，如果v超过了30mV，应该被设置为30mV。
        '''
        # dv = (self.a1 * self.v ** 2 + self.a2 * self.v + self.a3 - self.u + I) * dt
        # du = (self.a * (self.b * self.v - self.u)) * dt
        # self.v += dv
        # self.u += du
        # 改用四阶龙格库塔法
        k1_v = (self.a1 * self.v ** 2 + self.a2 * self.v + self.a3 - self.u + I)
        k1_u = self.a * (self.b * self.v - self.u)
        k2_v = (self.a1 * (self.v + 0.5 * dt * k1_v) ** 2 + self.a2 * (self.v + 0.5 * dt * k1_v) + self.a3 - self.u + I)
        k2_u = self.a * (self.b * (self.v + 0.5 * dt * k1_v) - self.u)
        k3_v = (self.a1 * (self.v + 0.5 * dt * k2_v) ** 2 + self.a2 * (self.v + 0.5 * dt * k2_v) + self.a3 - self.u + I)
        k3_u = self.a * (self.b * (self.v + 0.5 * dt * k2_v) - self.u)
        k4_v = (self.a1 * (self.v + dt * k3_v) ** 2 + self.a2 * (self.v + dt * k3_v) + self.a3 - self.u + I)
        k4_u = self.a * (self.b * (self.v + dt * k3_v) - self.u)
        self.v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6
        self.u += (k1_u + 2 * k2_u + 2 * k3_u + k4_u) * dt / 6
        self._generate_current(dt)
        return 30 if self.v > 30 else self.v

    def _generate_current(self, dt) -> float:
        if self.spike_current == -1:
            return -1
        if self.current_time - self.last_spike_time > 0.0 and \
            (self.current_time - self.last_spike_time) < 20:
            self.spike_current -= dt*self.time_factor*self.spike_current
        else:
            self.spike_current = 0.0
        return self.spike_current

    # def _generate_current(self, dt) -> float:
    #     if self.current_time - self.last_spike_time > 0.0 and \
    #         (self.current_time - self.last_spike_time) < 2:
    #         self.spike_current -= dt*self.time_factor*self.spike_current
    #     else:
    #         self.spike_current = -1.0
    #     return self.spike_current
        # if self.spike_current < -1:
        #     self.spike_current = -1
        #     return self.spike_current
        # if self.current_time - self.last_spike_time > 0.0 and \
        #         (self.current_time - self.last_spike_time) < 20:
        #     self.spike_current -= dt * self.time_factor * self.spike_current
        # else:
        #     self.spike_current = -1.0
        # return self.spike_current
class mizh_neuron(neuron):
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=2.0, refractory_time = 0.0):
        super().__init__(a=a, b=b, c=c, d=d, refractory_time = refractory_time)
        self.spike_current = -1
    def __mizh(self, v, u, I):
        temp = self.a1 * self.v ** 2 + self.a2 * self.v + self.a3
        vl = temp - u + I
        ul = self.a * (-temp - u)
        return vl, ul

    def step(self, I, dt=0.1):
        '''
                根据Izh博士的论文，如果v超过了30mV，应该被设置为30mV。
                '''
        # temp = self.a1 * self.v ** 2 + self.a2 * self.v + self.a3
        # dv = (temp - self.u + I) * dt
        # du = self.a*( -temp - self.u) * dt
        # self.v += dv
        # self.u += du
        # 改用四阶龙格库塔法
        # self.current_time += dt
        # if self.on_refractory:
        #     self.current_refractory_time -= dt
        #     I = 0.0
        #     if self.current_refractory_time <= 0:
        #         self.on_refractory = False

        I = super().step(I = I, dt = dt)

        k1_v, k1_u = self.__mizh(self.v, self.u, I)
        k2_v, k2_u = self.__mizh(self.v + 0.5 * dt * k1_v, self.u + 0.5 * dt * k1_u, I)
        k3_v, k3_u = self.__mizh(self.v + 0.5 * dt * k2_v, self.u + 0.5 * dt * k2_u, I)
        k4_v, k4_u = self.__mizh(self.v + dt * k3_v, self.u + dt * k3_u, I)
        self.v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6
        self.u += (k1_u + 2 * k2_u + 2 * k3_u + k4_u) * dt / 6
        self._generate_current(dt)
        return 30 if self.v > 30 else self.v

    def _generate_current(self, dt) -> float:
        if self.spike_current == -1:
            return -1
        if self.current_time - self.last_spike_time > 0.0 and \
            (self.current_time - self.last_spike_time) < 20:
            self.spike_current -= dt*self.time_factor*self.spike_current
        else:
            self.spike_current = 0.0
        return self.spike_current

class hh_neuron(neuron):
    '''
    参考自：https://blog.csdn.net/weixin_45834634/article/details/124434045
    '''
    def __init__(self,
                 V_rest: float = -70,
                 g_K_max=36, # inductance for K
                 g_Na_max=120, # inductance for Na
                 g_L: float = 0.3, # inductance
                 E_K: float = -12, #
                 E_Na: float = 115, #
                 E_L: float = 10.6, #
                 C_m: float = 1.0, # memberance capacity
                 refractory_time:float = 0.0
                 ):
        super().__init__(0, 0, V_rest, 0, refractory_time = refractory_time)

        self.V_rest = V_rest
        self.g_K_max = g_K_max
        self.g_Na_max = g_Na_max
        self.g_L = g_L
        self.E_K = E_K
        self.E_Na = E_Na
        self.E_L = E_L
        self.C_m = C_m

        self.v = 0.0
        self.last_v = self.v
        self.last_last_v = self.v
        self.alpha_m = 0.1 * ((25 - self.v) / (np.exp((25 - self.v) / 10) - 1))
        self.beta_m = 4 * np.exp(-1 * self.v / 18)
        self.alpha_n = 0.01 * ((10 - self.v) / (np.exp((10 - self.v) / 10) - 1))
        self.beta_n= 0.125 * np.exp(-1 * self.v / 80)
        self.alpha_h = 0.07 * np.exp(-1 * self.v / 20)
        self.beta_h = 1 / (np.exp((30 - self.v) / 10) + 1)

        self.m = self.alpha_m/(self.alpha_m+self.beta_m)
        self.n = self.alpha_n/(self.alpha_n+self.beta_n)
        self.h = self.alpha_h/(self.alpha_h+self.beta_h)

        self.time_factor = 0.05

    def reset(self):
        if self.on_refractory:
            return False
        if self.v + self.V_rest > 30:
            if True:
            # if self.last_v > self.v and self.last_last_v < self.last_v:
                self.on_refractory = True
                self.current_refractory_time = self.refractory_time
                self.spike_current = 1.0
                self.last_spike_time = self.current_time
            return True
        return False


    def threshold(self):
        super().threshold()

    def __dm_dt(self, m, alpha_m, beta_m):
        return alpha_m * (1 - m) - beta_m * m

    def __dn_dt(self, n, alpha_n, beta_n):
        return alpha_n * (1 - n) - beta_n * n

    def __dh_dt(self, h, alpha_h, beta_h):
        return alpha_h * (1 - h) - beta_h * h

    def __update_var(self, var, dt, alpha, beta, dvar_dt):
        k1 = dt * dvar_dt(var, alpha, beta)
        k2 = dt * dvar_dt(var + 0.5 * k1, alpha, beta)
        k3 = dt * dvar_dt(var + 0.5 * k2, alpha, beta)
        k4 = dt * dvar_dt(var + k3, alpha, beta)
        return var + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def step(self, I, dt) -> float:
        self.last_v = self.v
        self.last_last_v = self.last_v
        I = super().step(I = I, dt = dt)
        self.alpha_m = 0.1 * ((25 - self.v) / (np.exp((25 - self.v) / 10) - 1))
        self.beta_m = 4 * np.exp(-1 * self.v / 18)
        self.alpha_n = 0.01 * ((10 - self.v) / (np.exp((10 - self.v) / 10) - 1))
        self.beta_n= 0.125 * np.exp(-1 * self.v / 80)
        self.alpha_h = 0.07 * np.exp(-1 * self.v / 20)
        self.beta_h = 1 / (np.exp((30 - self.v) / 10) + 1)

        # 电导率
        self.g_Na = self.m ** 3 * self.g_Na_max * self.h
        self.g_K = self.n ** 4 * self.g_K_max

        I_Na = self.g_Na * (self.v - self.E_Na)
        I_K = self.g_K * (self.v - self.E_K)
        I_L = self.g_L * (self.v - self.E_L)
        I_ion = I - I_K - I_Na - I_L

        # self.m = self.m + (self.alpha_m * (1 - self.m) - self.beta_m * self.m) * dt
        # self.n = self.n + (self.alpha_n * (1 - self.n) - self.beta_n * self.n) * dt
        # self.h = self.h + (self.alpha_h * (1 - self.h) - self.beta_h * self.h) * dt
        # 龙格库塔
        self.m = self.__update_var(self.m, dt, self.alpha_m, self.beta_m, self.__dm_dt)
        self.n = self.__update_var(self.n, dt, self.alpha_n, self.beta_n, self.__dn_dt)
        self.h = self.__update_var(self.h, dt, self.alpha_h, self.beta_h, self.__dh_dt)

        self.v = self.v + I_ion / self.C_m * dt
        self._generate_current(dt)

        return self.v + self.V_rest

    def get_v(self):
        return self.v + self.V_rest


class lif_neuron(neuron):
    def __init__(self,
                 c:float=-62.0,
                El:float=-60.0,
                 gl:float=2.0e-1,
                 Cex:float=0.5e-0,
                 Vt:float = -58.0,
                 refractory_time = 0.1):
        super().__init__(0,0,c,0, refractory_time = refractory_time)
        self.El = El
        self.v = El
        self.gl = gl
        self.Cex = Cex
        self.Vt = Vt
        self.time_factor = 1

    def step(self, I, dt) -> float:
        I = super().step(I = I, dt = dt)
        dv = (self.gl * (self.El - self.v) + I)/self.Cex * dt
        self.v += dv
        # if self.v > self.Vt:
        #     self.spike_current = 1.0
        self._generate_current(dt)
        return self.v

    def threshold(self):
        if self.v >= self.Vt:
            self.v = self.c
            return True
        return False

    def reset(self):
        if self.v >= self.Vt:
            self.on_refractory = True
            self.current_refractory_time = self.refractory_time
            self.spike_current = self.max_current
            self.last_spike_time = self.current_time
            self.v = self.c
            return True
        return False


def simulate(n:neuron, dt = 0.1,t_max = 50, I_stimulus = 10.0, start_time = 0.0):
    max_interval = (int)(t_max / dt)
    time = np.arange(0, t_max, dt)
    voltage_trace = np.zeros(max_interval)
    I = np.zeros(max_interval)
    I_start_index = (int)(start_time / dt)
    I[I_start_index:] = I_stimulus

    for t in range(max_interval):
        n.reset()
        voltage_trace[t] = n.step(I[t], dt)
        # neuron.threshold()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, voltage_trace, color='black')
    plt.title(f'{n.customized_name} Neuron Model')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')

    plt.subplot(2, 1, 2)
    plt.plot(time, I, color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Current (mV)')
    plt.tight_layout()
    # plt.savefig(f'Memristive Izhikevich Neuron Model.eps', format='eps')
    plt.show()

def simulate_for_mizh():
    izh = mizh_neuron(c=-65)
    t_max = 300
    dt = 0.1
    max_interval = (int)(t_max / dt)
    time = np.arange(0, t_max, dt)
    voltage_trace = np.zeros(max_interval)
    I = np.zeros(max_interval)
    I[1000:] = 10
    for t in range(max_interval):
        # izh.threshold()
        izh.reset()
        voltage_trace[t] = izh.step(I[t], dt)


    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, voltage_trace)
    plt.title('Memristive Izhikevich Neuron Model')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')

    plt.subplot(2, 1, 2)
    plt.plot(time, I)
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Current (mV)')
    plt.tight_layout()
    plt.savefig(f'Memristive Izhikevich Neuron Model.eps', format='eps')
    plt.show()

RS = [(0.02, 0.2, -65, 8), 'RS']
IB = [(0.02, 0.2, -55, 4), 'IB']
CH = [(0.02, 0.2, -50, 2), 'CH']
FS = [(0.1, 0.2, -65, 2), 'FS']
LTS = [(0.02, 0.25, -65, 2), 'LTS']
# todo: TC and RZ.
# TC = [(0.02, 0.25, -65, 0.05), 'TC']
# RZ = [(0.1, 0.26, -65, 2), 'RZ']

def simulate_for_izh():
    '''
    RS: c = -85， d = 8
    IB: c = -55, d = 4
    CH: c = -50, d = 2
    FS: a = 0.1
    LTS: b = 0.25
    '''
    mode = CH # 在这里修改模式
    izh = izh_neuron(*(mode[0]))
    t_max = 300
    dt = 0.1
    max_interval = (int)(t_max / dt)
    time = np.arange(0, t_max, dt)
    voltage_trace = np.zeros(max_interval)
    I = np.zeros(max_interval)
    I[1000:] = 15
    izh.set_refractory(20)
    for t in range(max_interval):
        izh.reset()
        voltage_trace[t] = izh.step(I[t], 0.1)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, voltage_trace)
    plt.title(f'Izhikevich Neuron Model ({mode[1]})')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential ()')


    plt.subplot(2,1,2)
    plt.plot(time, I)
    # plt.title('Hodgkin Huxley Neuron Model')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Current (mV)')
    plt.tight_layout()
    # plt.savefig(f'Izhikevich Neuron Model ({mode[1]}).eps', format='eps')
    plt.show()

def simulate_for_hh(g_L: float = 0.3, # inductance
                    E_K: float = -12, #
                    E_Na: float = 115, #
                    E_L: float = 10.6, #
                    C_m: float = 1.0 # membrane capacity
                   ):
    izh = hh_neuron(g_L=g_L, E_K=E_K, E_Na=E_Na, E_L=E_L, C_m=C_m)
    t_max = 300
    dt = 0.01
    max_interval = int(t_max / dt)
    time = np.arange(0, t_max, dt)
    voltage_trace = np.zeros(max_interval)
    I = np.zeros(max_interval)
    # Set current input
    I[10000:] = 10
    izh.set_refractory(50)

    for t in range(max_interval):
        izh.reset()
        voltage_trace[t] = izh.step(I[t], dt)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Left Y-axis: Current
    ax1.plot(time, I, color='b', label='Input Current (nA)')
    ax1.set_ylabel('Input Current (nA)', color='b')
    ax1.set_ylim(0, 50)
    ax1.set_xlabel('Time (ms)')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')  # Legend for current

    # Right Y-axis: Voltage
    ax2 = ax1.twinx()
    ax2.plot(time, voltage_trace, color='r', label='Membrane Potential (mV)')
    ax2.set_ylabel('Membrane Potential (mV)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')  # Legend for voltage

    # Title and layout
    plt.title('Hodgkin Huxley Neuron Model')
    fig.tight_layout()
    plt.savefig('Hodgkin_Huxley_Neuron_Model.eps', format='eps')
    plt.show()

if __name__ == '__main__':
    '''default: 
    def simulate_for_hh( g_L: float = 0.3, # inductance
                 E_K: float = -12, # 
                 E_Na: float = 115, # 
                 E_L: float = 10.6, # 
                 C_m: float = 1.0 # memberance capacity
        ):'''
    # simulate_for_hh()
    # simulate_for_hh(g_L= 0.6)
    # simulate_for_hh(g_L = 0.15)
    # simulate_for_hh(E_K = -24)
    # simulate_for_hh(E_K = -6)
    # simulate_for_hh(E_Na = 60)
    # simulate_for_hh(E_Na = 240)
    # simulate_for_hh(C_m = 0.5)
    # simulate_for_hh(C_m = 2.0)
    # simulate_for_hh(C_m = 1e-1)
    # simulate_for_izh()
    izh = lif_neuron(c=-62.0,
                El=-60.0,
                 gl=2.0e-1,
                 Cex=0.5e-0,
                 Vt = -58.0,
                refractory_time=10)
    t_max = 300
    dt = 0.05
    max_interval = (int)(t_max / dt)
    time = np.arange(0, t_max, dt)
    voltage_trace = np.zeros(max_interval)
    I = np.zeros(max_interval)
    I[100:] = 0.5
    for t in range(max_interval):
        izh.reset()
        voltage_trace[t] = izh.step(I[t], dt)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, voltage_trace)
    plt.title(f'LIF Neuron Model')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential ()')
    plt.subplot(2,1,2)
    plt.plot(time, I)
    plt.title("Current")
    plt.show()


    # simulate_for_hh()
    # simulate_for_izh()
    # izh = izh_neuron(c=-65)
    # mizh = mizh_neuron(c=-65)
    # hh = hh_neuron()
    # t_max = 100
    # dt = 0.01
    # max_interval = (int)(t_max / dt)
    # time = np.arange(0, t_max, dt)
    # izh_voltage_trace = np.zeros(max_interval)
    # mizh_voltage_trace = np.zeros(max_interval)
    # hh_voltage_trace = np.zeros(max_interval)
    #
    # I = np.zeros(max_interval)
    # I[0:5000] = 20
    # for t in range(max_interval):
    #     izh.reset()
    #     mizh.reset()
    #     izh_voltage_trace[t] = izh.step(I[t], dt)
    #     mizh_voltage_trace[t] = mizh.step(I[t], dt)
    #
    #     hh_voltage_trace[t] = hh.step(I[t], dt)
    # plt.figure(figsize=(7.5, 7.5))
    # plt.subplot(3, 1, 1)
    # plt.plot(time, izh_voltage_trace, label='Izhikevich Neuron Model')
    # plt.title('Izhikevich Neuron Model')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Membrane Potential (mV)')
    # plt.subplot(3, 1, 2)
    # plt.plot(time, mizh_voltage_trace, label='Memristive Izhikevich Neuron Model')
    # plt.title('Memristive Izhikevich Neuron Model')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Membrane Potential (mV)')
    # plt.subplot(3, 1, 3)
    # plt.plot(time, hh_voltage_trace, label='Memristive Izhikevich Neuron Model')
    # plt.title('Hodgkin Huxley Neuron Model')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Membrane Potential (mV)')
    # plt.tight_layout()
    # plt.show()
