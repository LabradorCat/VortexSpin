import numpy as np
import os
import csv
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
plt.rcParams['animation.ffmpeg_path'] = os.path.join(os.getcwd(), r'ffmpeg\bin\ffmpeg.exe')

# Customized colormap for nanobars
clist1 = list('bkr')
clist1 = ['#ff0000' if c == 'r' else c for c in clist1]
clist1 = ['#0000ff' if c == 'b' else c for c in clist1]
clist1 = ['#000000' if c == 'k' else c for c in clist1]
mycmap = ListedColormap(clist1)
mynorm = BoundaryNorm([-1, -0.5, 0.5, 1], mycmap.N)


def test_cmap(cmap):
    """
    testing and showing color map
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = x
    c = np.linspace(-1, 1, 100)
    ax.scatter(x, y, s=50, c=c, cmap=cmap)
    plt.show()


def load_summary(folder, output):
    '''
        load in statistical summary from field sweep
    '''
    summary = None
    for root, dirs, files in os.walk(folder):
        sum_files = list(filter(lambda x: 'ASVIStateInfo' in x, files))
        for file in sum_files:
            if '.npz' not in file:
                file = file + '.npz'
            npzfile = np.load(os.path.join(root, file), allow_pickle=True)
            parameters = npzfile['arr_0']
            fieldloops = npzfile['arr_1']
            q = npzfile['arr_2']
            mag = npzfile['arr_3']
            monopole = npzfile['arr_4']
            vortex_count = npzfile['arr_5']
            macrospin_count = npzfile['arr_6']
            frequency = npzfile['arr_7']
            summary = {
                'parameters': np.asarray(parameters),
                'fieldloops': np.asarray(fieldloops),
                'q': np.asarray(q),
                'mag': np.asarray(mag),
                'monopole': np.asarray(monopole),
                'vortex_count': np.asarray(vortex_count),
                'macrospin_count': np.asarray(macrospin_count),
                'FMR_frequency': np.asarray(frequency)
            }.get(output, None)
    return summary


def plot_applied_field(field):
    # plotting external field
    fig, axes = plt.subplots(1, 2, sharex=True)
    fig.suptitle('Applied Field on the Lattice')
    ax1, ax2 = axes[0], axes[1]

    steps = np.arange(0, len(field), 1)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Applied Field (mT)')
    ax1.plot(steps, 1000 * field, 'o')

    field = field[field > 0]
    steps = 2 * np.arange(0, len(field), 1)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Applied Field (mT)')
    ax2.plot(steps, 1000 * field, 'o')
    ax2.plot(steps, 1000 * field, 'k--')
    plt.show()


def plot_vortex_macrospin_number(vortex_count, macrospin_count, fitfunc, ax=None):
    '''
    Plot the time evolution of vortex numbers and macrospin numbers
    '''
    assert len(vortex_count) == len(macrospin_count)
    assert 'exp' or 'sigmoid' in fitfunc

    if ax is None:
        fig, ax = plt.subplots()

    size = len(vortex_count)
    steps = np.arange(0, size, 1)

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    def sigmoid(x, a, b, c, d):
        return (a / (1 + np.exp(-b * x + c))) + d

    if fitfunc == 'exp':
        popt1, pcov1 = curve_fit(exp_decay, steps, vortex_count,
                                 p0=[-np.max(vortex_count), 1, np.max(vortex_count)])
        popt2, pcov2 = curve_fit(exp_decay, steps, macrospin_count,
                                 p0=[np.max(macrospin_count) / 2, 1, np.max(macrospin_count) / 2])
        ax.plot(steps, exp_decay(steps, *popt1), 'r--', zorder=2)
        ax.plot(steps, exp_decay(steps, *popt2), 'b--', zorder=2)

    if fitfunc == 'sigmoid':
        popt1, pcov1 = curve_fit(sigmoid, steps, vortex_count,
                                 p0=[np.min(vortex_count)-np.max(vortex_count), 1, 5, np.max(vortex_count)])
        popt2, pcov2 = curve_fit(sigmoid, steps, macrospin_count,
                                 p0=[np.max(macrospin_count)-np.min(macrospin_count), 1, 5, np.min(macrospin_count)])
        ax.plot(steps, sigmoid(steps, *popt1), 'r--', zorder=2)
        ax.plot(steps, sigmoid(steps, *popt2), 'b--', zorder=2)

    ax.set_title('Evolution of Vortex and Macrospin Modes')
    ax.set_xlabel('steps')
    ax.set_ylabel('number of vortices')
    ax.set_xlim(0, size + 10)
    ax.set_ylim(0, np.max(macrospin_count) + 5)
    ax.plot(steps, vortex_count, 'o', label='number of vortices (Nv)', zorder=1)
    ax.plot(steps, macrospin_count, 'o', label='number of macrospins (Nm)', zorder=1)
    ax.legend()
    ax.grid()


def plot_vector_field_2D(asvi, fig, ax, color=None):
    """
    Return a vector field according to the input lattice
    Represented elements include:
        - Macrospins: red/blue arrows
        - Vertices: colored dots with color representing vertex charges
        - Vortices: black dots
    Input parameter specifications:
        - lattice: 3D numpy.ndarray from the ASVI class
        - ax: the matplotlib axis to plot the vector field
    """
    # extracting useful values from lattice
    pos = asvi.pos_matrix()
    mag = asvi.mag_matrix()
    vc = asvi.vc_matrix()
    X = pos[:, :, 0].flatten()
    Y = pos[:, :, 1].flatten()
    Mx = mag[:, :, 0].flatten()
    My = mag[:, :, 1].flatten()
    Cv = vc.flatten()
    bar_l = asvi.get_attribute_matrix('bar_l').flatten()
    bar_w = asvi.get_attribute_matrix('bar_w').flatten()
    bar_w[np.isnan(bar_w)] = 0
    U = Mx * bar_l  # normalise vector
    V = My * bar_l  # normalise vector
    # sorting out colors and thicknesses
    line_w = 4 * (bar_w > 150e-9) + 1
    if color is None:
        line_rbg = np.arctan(Mx + My)
        color = mycmap(mynorm(line_rbg))
    # plotting vector field
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, pivot='mid', zorder=1,
              linewidths=line_w, color=color, edgecolors=color)
    #ax.scatter(X, Y, s=50, c=Cv, cmap='gist_rainbow', marker='o', zorder=2, vmax=1, vmin=-1)
    ax.set_xlabel('Lx (m)')
    ax.set_ylabel('Ly (m)')
    ax.ticklabel_format(style='sci', scilimits=(0, 0))
    ax.use_sticky_edges = False
    return ax


def FMR_specturm(data, folder, fmin=2, fmax=11, bins=396, bandwidth=0.01):

    def kde_sklearn(x, x_grid, bandwidth, **kwargs):
        """Kernel Density Estimation with Scikit-learn"""
        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        # score_samples() returns the log-likelihood of the samples
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return np.exp(log_pdf)
    # Setting color code
    num_plots = len(data)
    f_arr = np.linspace(fmin, fmax, bins, endpoint=False)

    FMR_data1 = []      # histogram data
    FMR_data2 = []      # kde data
    for i in tqdm(range(0, num_plots-10), desc='Spectrum analysis progress: ', unit='step'):
        h_app = data[i, 0]
        h_app2 = data[i, 1]
        freq = data[i, 2:]
        f_pdf = kde_sklearn(freq, f_arr, bandwidth=bandwidth)
        hist, bin_edge = np.histogram(freq, bins=bins, range=(fmin, fmax))
        FMR_data1.append(np.append([h_app, h_app2], hist))
        FMR_data2.append(np.append([h_app, h_app2], f_pdf))
    # Export FMR spectrum to excel file
    header = ['H_app', 'H_app2']
    for i in range(len(f_arr)):
        header.append(f'IQ{i}')
    df_hist = pd.DataFrame(data=FMR_data1, columns=header)
    df_kde = pd.DataFrame(data=FMR_data2, columns=header)
    # Write each dataframe to a different worksheet.
    with pd.ExcelWriter(folder, engine='xlsxwriter') as writer:
        df_hist.to_excel(writer, sheet_name='hist')
        df_kde.to_excel(writer, sheet_name=f'kde_bw{bandwidth}')
    print('Output Successful!')


def plot_FMR_spectrum(FMR_file, steps, fmin=2, fmax=10.5, bins=396, bandwidth=0.1):

    FMR_df1 = pd.read_excel(FMR_file, sheet_name='hist')
    FMR_arr1 = FMR_df1.to_numpy()
    freq = FMR_arr1[:, 3:]

    FMR_df2 = pd.read_excel(FMR_file, sheet_name=f'kde_bw{bandwidth}')
    FMR_arr2 = FMR_df2.to_numpy()
    f_pdf = FMR_arr2[:, 3:]
    f_arr = np.linspace(fmin, fmax, bins, endpoint=False)

    num_plots = len(f_arr)
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    ax1, ax2 = axes[0], axes[1]
    fig.suptitle('FMR Spectrum')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Occurrence')
    ax1.set_xlim(fmin, fmax)
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Probability')

    loop = 0
    for i in range(0, num_plots):
        if i == 0:
            ax1.plot(f_arr, freq[i], label='Initial', color='blue', linewidth=2, alpha=1)
            ax2.plot(f_arr, f_pdf[i], label='Initial', color='blue', linewidth=2, alpha=1)
        elif i == num_plots - 22:
            ax1.plot(f_arr, freq[i], label='Final', color='red', linewidth=2, alpha=1)
            ax2.plot(f_arr, f_pdf[i], label='Final', color='red', linewidth=2, alpha=1)
        # elif i % steps == 0:
        #     ax1.hist(freq, bins=bins, histtype='step', label=f'loop {loop}', linewidth=3, alpha=0.5)
        #     ax2.plot(f_arr, f_pdf, label=f'loop {loop}', linewidth=3, alpha=0.5)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')
        loop += 1


def FMR_animation(FMR_file, sheet_name, output_folder, fps=10, figsize=(8, 8)):
    print('STARTING TO MAKE FMR ANIMATION...')
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='FMR Simulation', artist='Matplotlib',
                    comment='FMR spectrum evolution animation')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    FMR_df = pd.read_excel(FMR_file, sheet_name=sheet_name)
    FMR_arr = FMR_df.to_numpy()
    spectrum = FMR_arr[:, 3:]

    with writer.saving(fig, (os.path.join(output_folder, "FMR_animation.mp4")), 100):
        for i, spec in enumerate(tqdm(spectrum, desc='FMR animation progress: ', unit='frame')):
            x = np.linspace(0, len(spec), len(spec), endpoint=False)
            spec = np.array(spec)
            ax.clear()
            ax.set_ylim(0, 2)
            ax.set_title('FMR Spectrum', loc='center', pad=20)
            ax.set_title(f"Steps: {i}", loc='left', pad=20)
            ax.plot(x, spec, 'k-', linewidth=2)
            writer.grab_frame()
    print('ANIMATION COMPLETE!')


def FMR_heatmap(type=0, field=0, bias=0, display_FMR_heatmap=False):
    if display_FMR_heatmap:
        uH = np.linspace(-10, 10, 1000)
        plt.figure('FMR_heatmap', figsize=(3,3))
        plt.xlabel('uH(mT)')
        plt.xlim(-10, 10)
        plt.ylabel('frequency (GHz)')
        plt.ylim(2, 11)
        plt.plot(uH, FMR_heatmap(0, uH))
        plt.plot(uH, FMR_heatmap(1, uH))
        plt.plot(uH, FMR_heatmap(2, uH))
        plt.grid()
        plt.show()
    else:
        freq = 0
        if type == 0:  # thin bar
            freq = -0.035 * field + 8.7
        if type == 1:  # thick bar
            freq = -0.035 * field + 6.9
        if type == 2:  # vortex
            freq = -1.86 * np.tanh(0.2 * field) + 4.5
        return np.random.normal(loc=freq*(1+bias), scale=0.1*(1-bias), size=None)


# if __name__ == '__main__':
#     folder = 'D:\ASI_MSci_Project\ASVI_Simulation_Output'
#     FMR_output = 'E:\ASI_MSci_Project\ASVI_Predictions'
#     FMR_file_name = 'sim_sin_100x100_20to25mT.xlsx'
#     output = os.path.join(FMR_output, FMR_file_name)
#     vc = load_summary(folder, output='vortex_count')
#     mc = load_summary(folder, output='macrospin_count')
#     fd = load_summary(folder, output='fieldloops')
#     FMR_f = load_summary(folder, output='FMR_frequency')
#     # plot_vortex_macrospin_number(vc, mc, 'exp')
#     # plot_applied_field(fd)
#     # FMR_specturm(FMR_f, output, plotting=False, steps=300, fmin=4.0, fmax=10.5, bins=396, bandwidth=0.1)
#     # FMR_animation(FMR_file=output, sheet_name='kde_bw0.1', output_folder=folder)
#     # plot_FMR_spectrum(output, steps=300)
