import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy.linalg
import collections
from tqdm import tqdm
# Vortex Spin Modules
from plotting import plot_vector_field_2D, FMR_heatmap
from asvi_materials import NanoBar, Vertex
from fields import *

plt.rcParams['animation.ffmpeg_path'] = os.path.join(os.getcwd(), r'ffmpeg\bin\ffmpeg.exe')


class ASVI:
    '''
    Artificial Spin Vortex Ice model to be performing field sweeps
    Supported lattice types:
        - Square
        - Square Staircase with thin-thick nanobars
    '''

    # INITIALISATION
    def __init__(self, unit_cells_x=25, unit_cells_y=25, vertex_gap=100e-9,
                 interType='dumbbell', periodicBC=False):
        # Lattice Parameters
        assert interType in ('dumbbell', 'dipole')
        self.unit_cells_x = unit_cells_x
        self.unit_cells_y = unit_cells_y
        self.vertex_gap = vertex_gap
        self.unit_cell_len = None
        self.side_len_x = None
        self.side_len_y = None
        self.lattice = None
        self.previous = None
        self.type = None
        # Simulation Parameters
        self.interType = interType
        self.periodicBC = periodicBC
        self.field_steps = None
        self.field_angle = None
        self.applied_field = None
        self.Hmax = None
        self.Hmin = None
        self.steps = None

    # LATTICE PROPERTIES GETTING & SETTING
    def returnLattice(self):
        '''
        Returns the lattice in its current state
        '''
        return (self.lattice)

    def clearLattice(self):
        '''
        Clears the lattice
        '''
        self.lattice = None

    def get_attribute_matrix(self, attr, grid=None):

        assert type(attr) == str
        if grid is None:
            grid = self.lattice
        len_x = grid.shape[0]
        len_y = grid.shape[1]

        attr_matrix = []
        for x in range(0, len_x):
            attr_matrix.append([])
            for y in range(0, len_y):
                obj = grid[x, y]
                if hasattr(obj, attr):
                    attr_matrix[x].append(getattr(obj, attr))
                else:
                    attr_matrix[x].append(None)
        return np.asarray(attr_matrix, dtype=float)

    def pos_matrix(self, grid=None):
        return self.get_attribute_matrix('pos', grid)

    def mag_matrix(self, grid=None):
        return self.get_attribute_matrix('mag', grid)

    def hc_matrix(self, grid=None):
        return self.get_attribute_matrix('hc', grid)

    def vc_matrix(self, grid=None):
        return self.get_attribute_matrix('v_c', grid)

    # SAVE & LOAD FUNCTIONS
    def save(self, file, folder=os.getcwd()):
        '''
        Save existing arrays
        '''
        if folder == None:
            folder = os.getcwd()
        else:
            if not os.path.exists(folder):
                os.makedirs(folder)
        file = file.replace('.', 'p')
        parameters = np.array([self.unit_cells_x, self.unit_cells_y, self.vertex_gap,
                               self.side_len_x, self.side_len_y, self.applied_field, self.field_angle])
        np.savez_compressed(os.path.join(folder, file), self.lattice, parameters)

    def load(self, file):
        '''
        load in existing arrays
        '''
        if '.npz' not in file:
            file = file + '.npz'
        npzfile = np.load(file, allow_pickle=True)
        parameters = npzfile['arr_1']
        self.unit_cells_x = np.int(parameters[0])
        self.unit_cells_y = np.int(parameters[1])
        self.vertex_gap = np.float(parameters[2])
        self.side_len_x = np.int(parameters[3])
        self.side_len_y = np.int(parameters[4])
        self.applied_field = np.float(parameters[5])
        self.field_angle = np.float(parameters[6])
        self.lattice = npzfile['arr_0']

    # LATTICE TYPES
    def square(self, hc_m=0.03, hc_v=0.02, hc_std=0.05, hc_v_std=0.03,
               magnetisation=800e3, bar_l=600e-9, bar_w=125e-9, bar_t=20e-9):
        '''
        Defines the lattice positions, magnetisation directions and coercive fields of an array of
        square ASI
        Takes the unit cell from the initial defined parameters
        Generates a normally distributed range of coercive fields of the bars using Hc_mean and Hc_std as a percentage
        One thing to potentially change is to have the positions in nanometers
        '''
        self.type = 'square'
        self.side_len_x = 2 * self.unit_cells_x + 1
        self.side_len_y = 2 * self.unit_cells_y + 1
        self.unit_cell_len = (bar_l + self.vertex_gap) / 2
        grid = []
        for x in range(0, self.side_len_x):
            grid.append([])
            for y in range(0, self.side_len_y):
                xpos = x * self.unit_cell_len
                ypos = y * self.unit_cell_len
                if (x + y) % 2 != 0:
                    if y % 2 == 0:
                        grid[x].append(NanoBar(xpos, ypos, 0, 1, 0, 0, hc_m, hc_v, hc_std, hc_v_std,
                                               bar_l, bar_w, bar_t, magnetisation, 'macrospin'))
                    else:
                        grid[x].append(NanoBar(xpos, ypos, 0, 0, 1, 0, hc_m, hc_v, hc_std, hc_v_std,
                                               bar_l, bar_w, bar_t, magnetisation, 'macrospin'))
                else:
                    if x % 2 == 0 and x != 0 and y != 0 and x != self.side_len_x - 1 and y != self.side_len_x - 1:
                        grid[x].append(Vertex(xpos, ypos, 0, v_c=0.))
                    else:
                        grid[x].append(Vertex(xpos, ypos, 0, v_c=None))     # pseudo-vertex only serve as place-holders
        self.lattice = np.asarray(grid, dtype=object)

    def square_staircase(self, hc_thin=0.03, hc_thick=0.015, hc_v=0.02, hc_std=0.05, hc_v_std=0.03, magnetisation=800e3,
                         bar_l=600e-9, thin_bar_w=125e-9, thick_bar_w=200e-9, bar_t=20e-9):
        self.square(hc_thin, hc_v, hc_std, hc_v_std, magnetisation, bar_l, thin_bar_w, bar_t)
        grid = copy.deepcopy(self.lattice)
        for x in range(0, self.side_len_x):
            for y in range(0, self.side_len_y):
                obj = grid[x, y]
                if x % 4 == 0 and y % 4 == 1:
                    obj.bar_w = thick_bar_w
                    obj.set_hc(hc_thick, hc_std)
                elif x % 4 == 1 and y % 4 == 2:
                    obj.bar_w = thick_bar_w
                    obj.set_hc(hc_thick, hc_std)
                elif x % 4 == 2 and y % 4 == 3:
                    obj.bar_w = thick_bar_w
                    obj.set_hc(hc_thick, hc_std)
                elif x % 4 == 3 and y % 4 == 0:
                    obj.bar_w = thick_bar_w
                    obj.set_hc(hc_thick, hc_std)
        self.lattice = grid

    # SIMULATION EXECUTABLES
    def fieldSelect(self, fieldType, steps, Hmax, Hmin=None, Htheta=45):
        if fieldType == 'Sine':
            if Hmin is None:
                Hmin = - Hmax
            self.field_steps = Sine(steps, Hmax, Hmin)
        elif fieldType == 'Adaptive':
            if Hmin is None:
                Hmin = np.nanmin(self.hc_matrix())
            self.field_steps = Adaptive(steps, Hmax, Hmin)
        elif fieldType == 'Linear':
            if Hmin is None:
                Hmin = - Hmax
            self.field_steps = Linear(steps, Hmax, Hmin)
        elif fieldType == 'Sine_Train':
            if Hmin is None:
                Hmin = np.nanmin(self.hc_matrix())
            self.field_steps = Sine_Train(steps, Hmax, Hmin)
        elif fieldType == 'MackeyGlass':
            if Hmin is None:
                Hmin = np.nanmin(self.hc_matrix())
            self.field_steps = MackeyGlass_Train(steps, Hmax, Hmin)
        elif fieldType == 'MackeyGlass_exp':
            self.field_steps = MackeyGlass_exp()
        elif fieldType == 'Sine_exp':
            self.field_steps = Sine_exp()
        else:
            raise TypeError('fieldType not included!')
        self.Hmin = Hmin
        self.Hmax = Hmax
        self.field_angle = Htheta
        self.steps = len(self.field_steps)

    def fieldSweep(self, n=1, loops=1, folder=None, FMR=False, FMR_step=2, FMR_field=None):
        """
        Sweeps through the lattice using the designated field type.
        Total number of steps for a full minor loop is (2 * step).
        Allowed FieldTypes are:
            Sine, Adaptive
        The function then performs loops number of minor loops
        The Lattice after each field step gets saved to a folder. if folder is None then the
        function saves the lattice to the current working directory
        """
        # Determine which field type to sweep the lattice

        # Working out field angle and amend field steps
        Hrad = np.deg2rad(self.field_angle)
        Hmax, steps, Htheta = self.Hmax, self.steps, self.field_angle
        if np.sin(Hrad) == 0:
            angleFactor = np.cos(Hrad)
        else:
            angleFactor = np.sin(Hrad)
        # Create statistical parameters
        q, mag, monopole, fieldloops, frequency, vortex_count, macrospin_count = ([] for i in range(7))
        counter = 1
        i = 0
        # Start the field sweep
        print('STARTING SIMULATION WITH {} MINOR LOOPS...'.format(loops))
        self.relax(n=n)
        self.save('InitialASVI_Hmax{:.3f}_steps{}_Angle{:.0f}_n{}_Loops{}'.format(
            Hmax, steps, Htheta, n, loops), folder=folder)
        while i <= loops:
            self.previous = copy.deepcopy(self)
            for field in tqdm(self.field_steps, desc='Simulation Loop {} Progress: '.format(i), unit='step'):
                self.applied_field = field
                f_exp = field / angleFactor
                Happlied = f_exp * np.array([np.cos(Hrad), np.sin(Hrad), 0.])
                self.relax(Happlied, n)
                # applying FMR measurements
                if FMR and (counter % FMR_step) == 0:
                    if FMR_field is None:
                        h_app = Happlied
                        h_app2 = f_exp
                    else:
                        h_app = FMR_field * np.array([np.cos(Hrad), np.sin(Hrad), 0.])
                        h_app2 = FMR_field
                    freq = self.FMR_HM(h_app=h_app)
                    frequency.append(np.append([f_exp, h_app2], freq))
                # saving statistical data
                q.append(self.correlation(self.previous, self))
                mag.append(self.netMagnetisation())
                monopole.append(self.monopoleDensity())
                fieldloops.append(field)
                vortex_count.append(self.count_vortex())
                macrospin_count.append(self.count_macrospin())
                self.save('ASVIcounter{}_Loop{}_FieldApplied{:.3f}_Angle{:.0f}'.format(
                    counter, i, field, Htheta), folder=folder)
                counter += 1
            i += 1
        self.save('FinalASVI_Hmax{:.3f}_steps{}_Angle{:.0f}_n{}_Loops{}'.format(
            Hmax, steps, Htheta, n, loops), folder=folder)
        # Saving statistical information
        file = 'ASVIStateInfo_Hmax{:.3f}_steps{}_Angle{:.0f}_n{}_Loops{}'.format(Hmax, steps, Htheta, n, loops)
        parameters = np.array([Hmax, steps, Htheta, n, loops])
        if folder == None:
            folder = os.getcwd()
        np.savez(os.path.join(folder, file), parameters, fieldloops, q, mag, monopole, vortex_count, macrospin_count,
                 frequency)
        print('SIMULATION COMPLETE!')

    def fieldSweepAnimation(self, folder, figsize=(8, 8), fps=10):
        '''
        Will produce an animation of the lattice as it goes through the field sweep
        just provide the folder where the field sweeps are saved
        '''
        print('STARTING TO MAKE ANIMATION...')
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='ASVI Simulation', artist='Matplotlib',
                        comment='Artificial Spin Vortex Ice Simulation')
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        def sortFunc(element):
            begin = element.find('counter') + 7
            end = element.find('_Loop')
            return (int(element[begin:end]))

        with writer.saving(fig, (os.path.join(folder, "animation.mp4")), 100):
            for root, dirs, files in os.walk(folder):
                new_files = list(filter(lambda x: 'ASVIcounter' in x, files))
                new_files.sort(key=sortFunc)
                for file in tqdm(new_files, desc='Animation Progress: ', unit='frame'):
                    ax.clear()
                    self.clearLattice()
                    self.load(os.path.join(root, file))
                    self.vertexCharge()
                    # plotting vector field
                    ax = plot_vector_field_2D(self, fig, ax)
                    # setting title
                    H_applied = np.round(1000 * self.applied_field, 2)
                    H_theta = self.field_angle
                    steps = file[file.find('counter') + 7:file.find(r'_Loop')]
                    ax.set_title("Steps: " + steps, loc='left', pad=20)
                    ax.set_title('Applied Field: {} mT, Field Angle = {} deg'.format(H_applied, H_theta),
                                 loc='right', pad=20)
                    writer.grab_frame()
        print('ANIMATION COMPLETE!')

    # CALCULATIONS
    def relax(self, Happlied=None, n=1):
        '''
        Steps through all the the positions in the lattice and if the field applied along the direction
        of the bar is negative and greater than the coercive field then it switches the magnetisation
        of the bar
        '''
        if Happlied is None:
            Happlied = np.array([0, 0, 0], dtype=float)

        grid = copy.deepcopy(self.lattice)
        Xpos, Ypos = np.nonzero(grid)
        positions = np.array(list(zip(Xpos, Ypos)))
        positions_new = np.random.permutation(positions)
        for pos in positions_new:
            x = pos[0]
            y = pos[1]
            obj = grid[x, y]
            if obj.type == 'macrospin':  # test if object is a macrospin
                obj.h_local = self.Hlocal(x, y, n=n)
                unit_vector = obj.unit_vector
                field = np.dot(np.array(Happlied + obj.h_local), unit_vector)
                if field < -obj.hc:
                    obj.flip()
                    if np.random.random() <= self.vortex_prob(x, y):
                        obj.set_vortex()
            elif obj.type == 'vortex':  # test if object is a vortex
                obj.h_local = self.Hlocal(x, y, n=n)
                unit_vector = obj.unit_vector
                field = np.dot(np.array(Happlied + obj.h_local), unit_vector)
                if np.absolute(field) > obj.hc:
                    if np.linalg.norm(np.dot(obj.unit_vector, Happlied)) < 0:
                        obj.flip()
                    obj.set_macrospin()
        self.lattice = grid

    def Hlocal(self, x, y, n=1):
        """
        calculates the local field at position x, y including the
        field with n radius with n=1 just including nearest neighbours
        """
        grid = self.lattice
        obj = grid[x, y]
        r0 = obj.pos
        x1 = x - n
        x2 = x + n + 1
        y1 = y - n
        y2 = y + n + 1

        Hl = []
        if self.periodicBC:
            indx = np.arange(x1, x2)
            indy = np.arange(y1, y2)
            pos_matrix = np.take(self.pos_matrix(), indx, axis=0, mode='wrap')
            pos_matrix1 = np.take(pos_matrix, indy, axis=1, mode='wrap')
            mag_matrix = np.take(self.mag_matrix(), indx, axis=0, mode='wrap')
            mag_matrix1 = np.take(mag_matrix, indy, axis=1, mode='wrap')
            m = mag_matrix1.reshape(-1, mag_matrix1.shape[-1])
            r = pos_matrix1.reshape(-1, pos_matrix1.shape[-1])

            for pos, mag in zip(r, m):
                if np.linalg.norm(pos[0] - r0[0]) / (self.unit_cell_len * (n + 1)) > 1.0:
                    if pos[0] - r0[0] < 0:
                        pos[0] = self.side_len_x * self.unit_cell_len + pos[0]
                    else:
                        pos[0] = pos[0] - self.side_len_x * self.unit_cell_len

                if np.linalg.norm(pos[1] - r0[1]) / (self.unit_cell_len * (n + 1)) > 1.0:
                    if pos[1] - r0[1] < 0:
                        pos[1] = self.side_len_y * self.unit_cell_len + pos[1]
                    else:
                        pos[1] = pos[1] - self.side_len_y * self.unit_cell_len
                if not np.array_equal(pos, r0):
                    Hl.append(self.fieldCalc(x, y, mag, r0, pos))
            return sum(Hl)
        else:
            x1 = x - n
            x2 = x + n + 1
            y1 = y - n
            y2 = y + n + 1
            if x1 < 0:
                x1 = 0
            if x2 > self.side_len_x:
                x2 = self.side_len_x - 1
            if y1 < 0:
                y1 = 0
            if y2 > self.side_len_y - 1:
                y2 = self.side_len_y - 1

            for xi in range(x1, x2):
                for yi in range(y1, y2):
                    obji = grid[xi, yi]
                    pos = obji.pos
                    mag = obji.mag
                    if np.linalg.norm(pos - r0) / (n + 1) <= 1.0 and not np.array_equal(pos, r0):
                        field = self.fieldCalc(x, y, mag, r0, pos)
                        Hl.append(field)
            return sum(Hl)

    def fieldCalc(self, x, y, mag, r0, pos):
        '''
        Tells the class what type of field calculation method
        you want to use for the rest of the simulation
        '''
        obj = self.lattice[x, y]
        bar_length = obj.bar_l
        bar_width = obj.bar_w
        bar_thickness = obj.bar_t
        magnetisation = obj.magnetisation
        if self.interType == 'dipole':
            # Calculate a field in point r created by a dipole moment m located in r0.
            # Spatial components are the outermost axis of r and returned B.
            mag = np.array(mag)
            r0 = np.array(r0)
            pos = np.array(pos)
            mag = self.magnetisation * bar_length * bar_width * bar_thickness * mag
            # we use np.subtract to allow r0 and pos to be a python lists, not only np.array
            R = np.subtract(np.transpose(r0), pos).T
            # assume that the spatial components of r are the outermost axis
            norm_R = np.sqrt(np.einsum("i...,i...", R, R))
            # calculate the dot product only for the outermost axis,
            # that is the spatial components
            m_dot_R = np.tensordot(mag, R, axes=1)
            # tensordot with axes=0 does a general outer product - we want no sum
            B = 1e-7 * 3 * m_dot_R * R / norm_R ** 5 - np.tensordot(mag, 1 / norm_R ** 3, axes=0)
            # include the physical constant
            return B
        if self.interType == 'dumbbell':
            # Using the dumbbell model to calculate the interaction between each bar
            mag = np.array(mag)
            r0 = np.array(r0)
            pos = np.array(pos)
            mag_charge = bar_thickness * magnetisation * bar_width
            r2 = np.subtract(np.transpose(r0), pos).T + mag * bar_length / 2
            r1 = np.subtract(np.transpose(r0), pos).T - mag * bar_length / 2
            B = 1e-7 * mag_charge * (r1 / np.linalg.norm(r1) ** 3 - r2 / np.linalg.norm(r2) ** 3)
            return B

    def vertexCharge(self):
        '''
        Works you the vertex charge for square and Kagome.
        Should work on tetris and shakti but haven't test it yet
        '''
        grid = copy.deepcopy(self.lattice)
        for x in range(0, self.side_len_x):
            for y in range(0, self.side_len_y):
                obj = grid[x, y]
                if type(obj) == Vertex and obj.v_c is not None:
                    x1 = x - 1
                    x2 = x + 2
                    y1 = y - 1
                    y2 = y + 2

                    if x1 < 0:
                        x1 = 0
                    if x2 > self.side_len_x:
                        x2 = self.side_len_x
                    if y1 < 0:
                        y1 = 0
                    if y2 > self.side_len_y:
                        y2 = self.side_len_y
                    local = grid[x1:x2, y1:y2]
                    mag_matrix = self.mag_matrix(local)
                    net_charge = -(np.sum(mag_matrix[0:2, 0:2]) - np.sum(mag_matrix[1:3, 1:3]))
                    macro_count = self.count_macrospin(local)
                    if macro_count == 0:
                        charge = 0
                    else:
                        charge = net_charge / macro_count

                    obj.v_c = charge

        self.lattice = grid

    def FMR_HM(self, h_app=None):
        if h_app is None:
            h_app = [0, 0, 0]
        h_app = np.array([h_app])
        grid = copy.deepcopy(self.lattice)
        Xpos, Ypos = np.nonzero(grid)
        positions = np.array(list(zip(Xpos, Ypos)))

        freq = np.array([])
        for pos in positions:
            x = pos[0]
            y = pos[1]
            obj = grid[x, y]
            if type(obj) == NanoBar:
                if obj.type == 'vortex':
                    tp = 2
                elif obj.bar_w > 150e-9:
                    tp = 1
                else:
                    tp = 0
                B = 1000 * np.dot(np.array(h_app + obj.h_local), obj.unit_vector)  # convert to mT
                frequency = FMR_heatmap(type=tp, field=B, bias=obj.hc_bias)
                freq = np.append(freq, frequency)
        return np.array(freq)

    # STATISTICS
    def count_vortex(self, lattice=None):
        '''
        Count the number of vortices in the lattice
        '''
        if lattice is None:
            lattice = self.lattice
        count = 0
        xpos, ypos = np.nonzero(lattice)
        positions = np.array(list(zip(xpos, ypos)))
        for pos in positions:
            obj = lattice[pos[0], pos[1]]
            if obj.type == 'vortex':
                count += 1
        return count

    def count_macrospin(self, lattice=None):
        '''
            Count the number of vortices in the lattice
        '''
        if lattice is None:
            lattice = self.lattice
        count = 0
        xpos, ypos = np.nonzero(lattice)
        positions = np.array(list(zip(xpos, ypos)))
        for pos in positions:
            obj = lattice[pos[0], pos[1]]
            if obj.type == 'macrospin':
                count += 1
        return count

    def vortex_prob(self, x, y, v_n=1):
        x1 = x - v_n
        x2 = x + v_n + 1
        y1 = y - v_n
        y2 = y + v_n + 1
        if x1 < 0:
            x1 = 0
        if x2 > self.side_len_x:
            x2 = self.side_len_x
        if y1 < 0:
            y1 = 0
        if y2 > self.side_len_y:
            y2 = self.side_len_y

        grid = self.lattice
        neighbour = grid[x1:x2, y1:y2]
        bar_width = grid[x, y].bar_w
        min_width = 100e-9
        vortex_prob = 0
        if bar_width > min_width:  # thin bar below min_width cannot form vortex
            vortex_count = self.count_vortex(neighbour)
            if self.applied_field < 0:
                vortex_prob = 0.02 * vortex_count + 0.0305  # slightly more likely for vortex to from beside vortices
            else:
                vortex_prob = 0.02 * vortex_count + 0.0134
        return vortex_prob

    def correlation(self, lattice1, lattice2):
        '''
        Returns the correlation between lattice1 and lattice2
        '''

        l1 = lattice1.returnLattice()
        l2 = lattice2.returnLattice()
        total = 0
        same = 0
        for x in range(0, self.side_len_x):
            for y in range(0, self.side_len_y):
                obj1 = l1[x, y]
                obj2 = l2[x, y]
                if obj1.hc != 0:
                    if np.array_equal(obj1.mag, obj2.mag):
                        same += 1.0
                    total += 1.0
        return (same / total)

    def netMagnetisation(self):
        '''
        returns the magnetisation in the x and y directions
        '''
        grid = copy.deepcopy(self.lattice)
        mx, my, mz = (np.array([], dtype=float) for i in range(3))
        xpos, ypos = np.nonzero(grid)
        positions = np.array(list(zip(xpos, ypos)))
        for pos in positions:
            obj = grid[pos[0], pos[1]]
            if type(obj) == NanoBar:
                mx = np.append(mx, obj.mag[0])
                my = np.append(my, obj.mag[1])
                mz = np.append(mz, obj.mag[2])
        return np.array([np.nanmean(mx), np.nanmean(my), np.nanmean(mz)])

    def monopoleDensity(self):
        '''
        Returns the monopole density of a square or kagome lattice
        '''
        # 4in/0out have a charge of 1
        # 3in/1out have a charge of 0.5
        #   The density is then calculated by dividing by the total area minus the edges
        self.vertexCharge()
        grid = self.lattice
        magcharge = self.vc_matrix(grid).flatten()
        return (np.nanmean(np.absolute(magcharge)))



