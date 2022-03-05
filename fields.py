import os
import numpy as np
import pandas as pd
import collections

data_folder = os.path.join(os.getcwd(), 'experiments')
H_app_df = pd.read_excel(os.path.join(data_folder, 'fields.xlsx'))

def MackeyGlass_exp():
    samples = H_app_df['Mackey Glass'].to_numpy()
    field_steps = np.array([])
    for f in samples:
        field_steps = np.append(field_steps, [f, -f])
    return field_steps / 10000

def Adaptive(steps, Hmax, Hmin):
    '''
    Return array of field steps ranging from minimum Coercive field to Hmax
    Applied fields go from positive to negative
    (2 * steps) field values in a period
    '''
    field_steps = np.linspace(Hmin, Hmax, steps)
    field_steps = np.append(field_steps, np.negative(field_steps))
    return field_steps

def Sine(steps, Hmax, Hmin):
    '''
    return an array of field sweep values from Hmin to Hmax in the form of a sine wave
    (2 * steps) field values in a period
    '''
    field_steps = Hmax * np.sin(np.linspace(0, 2 * np.pi, 2 * steps))
    offset = Hmax - (Hmax - Hmin) / 2
    return field_steps + offset

def Linear(steps, Hmax, Hmin):
    '''
    Return a array of field steps linearly increasing field from Hmin to Hmax
    (steps) number of field values
    '''
    field_steps = np.linspace(Hmin, Hmax, steps)
    field_steps = np.negative(field_steps)
    return field_steps

def Sine_Train(steps, Hmax, Hmin):
    amp = (Hmax - Hmin) / 2
    offset = Hmax - amp
    steps = np.linspace(0, 2 * np.pi, 2 * steps, endpoint=False)
    field_steps = np.array([])
    for i in steps:
        field = amp * np.sin(i) + offset
        field_steps = np.append(field_steps, [field, -field])
    return field_steps

def MackeyGlass_Train(steps, Hmax, Hmin, tau=17, seed=None, n_samples=1):
    '''
    Generate the Mackey Glass time-series. Parameters are:
        - step: length of the time-series in timesteps, larger than 200 for best performance
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild
            chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
            timeseries at each invocation.
        - n_samples : number of samples to generate
    '''
    delta_t = 10
    history_len = tau * delta_t
    # Initial conditions for the history of the system
    timeseries = 1.2

    if seed is not None:
        np.random.seed(seed)
    samples = []
    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((steps, 1))
        for timestep in range(steps):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - 0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries
            # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples.append(inp)
        samples = np.array(samples).flatten()
        # Setting boundary and amplitude
        amp = (Hmax - Hmin) / 2
        off = Hmax - 0.75 * amp
        samples *= 2.5 * amp
        samples += off

    field_steps = np.array([])
    for f in samples:
        field_steps = np.append(field_steps, [f, -f])
    return field_steps
