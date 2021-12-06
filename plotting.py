import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_summary(file, output):
    '''
        load in statistical summary from field sweep
    '''
    assert 'RPMStateInfo' in file   # only allow summary file input

    if '.npz' not in file:
        file = file + '.npz'
    npzfile = np.load(file, allow_pickle = True)
    parameters = npzfile['arr_0']
    fieldloops = npzfile['arr_1']
    q = npzfile['arr_2']
    mag = npzfile['arr_3']
    monopole = npzfile['arr_4']
    vortex_count = npzfile['arr_5']
    macrospin_count = npzfile['arr_6']
    summary = {
        'parameters': parameters,
        'fieldloops': fieldloops,
        'q': q,
        'mag': mag,
        'monopole': monopole,
        'vortex_count': vortex_count,
        'macrospin_count': macrospin_count
    }.get(output, None)
    return summary

def plot_applied_field(field):
    # plotting external field
    steps = np.arange(0, len(field), 1)
    fig, ax = plt.subplots()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Applied Field (mT)')
    ax.set_title('Applied Field on the Lattice')
    ax.plot(steps, 1000 * field, 'k-')
    # ax.plot(int(file[file.find('counter') + 7:file.find(r'_Loop')]), H_applied,
    #          marker='o', markersize=3, markerfacecolor='red', markeredgecolor='red')
    plt.show()

def plot_vortex_macrospin_number(vortex_count, macrospin_count):
    '''
    Plot the time evolution of vortex numbers and macrospin numbers
    '''
    assert len(vortex_count) == len(macrospin_count)

    size = len(vortex_count)
    steps = np.arange(0, size, 1)
    def exp_dec(x, a, b):
        return a * (1 - np.exp(-b * x))
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt1, pcov1 = curve_fit(exp_dec, steps, vortex_count, p0=[np.max(vortex_count), 1])
    popt2, pcov2 = curve_fit(exp_decay, steps, macrospin_count, p0=[np.max(macrospin_count)/2, 1, np.max(macrospin_count)/2])

    fig, ax = plt.subplots()
    ax.set_title('Evolution of Vortex and Macrospin Modes')
    ax.set_xlabel('steps')
    ax.set_ylabel('number of vortices')
    ax.set_xlim(0, size + 10)
    ax.set_ylim(0, np.max(macrospin_count) + 5)
    ax.plot(steps, vortex_count, 'o', label = 'number of vortices (Nv)')
    ax.plot(steps, exp_dec(steps, *popt1), 'r--', label = 'Nv fitted')
    ax.plot(steps, macrospin_count, 'o', label = 'number of macrospins (Nm)')
    ax.plot(steps, exp_decay(steps, *popt2), 'b--', label = 'Nm fitted')
    ax.legend()
    ax.grid()
    plt.show()

if __name__ == '__main__':
    file = 'E:\ASI_MSci_Project\ASVI_Simulation_Output\RPMStateInfo_Hmax2.250000e-02_steps10_Angle7.853982e-01_neighbours2_Loops30'
    vc = load_summary(file, output='vortex_count')
    mc = load_summary(file, output='macrospin_count')
    fd = load_summary(file, output='fieldloops')
    plot_applied_field(fd)
    plot_vortex_macrospin_number(vc,mc)
