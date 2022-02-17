import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy.linalg
from tqdm import tqdm
from plotting import plot_vector_field_2D, FMR_heatmap

plt.rcParams['animation.ffmpeg_path'] = r'D:\ASI_MSci_Project\ffmpeg\bin\ffmpeg.exe'


class ASVI:
    '''
    Artificial Spin Vortex Ice model to be performing field sweeps

        Supported lattice types:
           - Square
           - Staircase Thin-Thick square

       The lattice is stored as a numpy array obeying the following index table
            INDEX       PROPERTIES
            0           x_pos
            1           y_pos
            2           z_pos
            3           x_mag
            4           y_mag
            5           z_mag
            6           Hc
            7           bar_length
            8           bar_width
            9           bar_thickness
            10          vertex Charge (0 for vertices, None for others)
            11          is Vortex (0 for dipolar nanobars, 1 for vortices, None for vertices)
            12          flip_count
            13          vortex_count
    '''

    # INITIALISATION
    def __init__(self, unit_cells_x=25, unit_cells_y=25, vertex_gap=1e-7,
                 bar_length=220e-9, bar_thickness=25e-9, bar_width=80e-9,
                 magnetisation=800e3):
        # Lattice Parameters
        self.unit_cells_x = unit_cells_x
        self.unit_cells_y = unit_cells_y
        self.unit_cell_len = (bar_length + vertex_gap) / 2
        self.side_len_x = None  # The side length is now defined in the square lattice
        self.side_len_y = None
        self.previous = None
        self.type = None
        self.Hc = None
        self.Hc_thick = None
        self.Hc_vortex = None
        self.Hc_std = None
        self.vertex_gap = vertex_gap
        self.bar_length = bar_length
        self.bar_width = bar_width
        self.bar_thickness = bar_thickness
        self.magnetisation = magnetisation
        # Simulation Parameters
        self.interType = 'dumbbell'
        self.periodicBC = False
        self.applied_field = None
        self.field_angle = None

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

    def get_bar_length(self, x, y):
        return  self.lattice[x, y, 7]

    def get_bar_width(self, x, y):
        return self.lattice[x, y, 8]

    def get_bar_thickness(self, x, y):
        return self.lattice[x, y, 9]

    def set_vortex(self, x, y, lattice = None):
        '''
        Set a point in lattice to vortex
        '''
        Hc_vortex = 0.026
        Hc_std = 0.01
        if lattice is None:
            lattice = self.lattice
        lattice[x, y, 11] = 1
        lattice[x, y, 3:6] = 0
        lattice[x, y, 6] = np.random.normal(loc=Hc_vortex, scale=Hc_std * Hc_vortex, size=None)
        return lattice

    def set_macrospin(self, x, y, lattice = None):
        '''
            Set a point in lattice to macrospin
        '''
        Hc = self.Hc_thick
        Hc_std = self.Hc_std
        if lattice is None:
            lattice = self.lattice
        lattice[x, y, 11] = 0
        lattice[x, y, 6] = np.random.normal(loc = Hc, scale = Hc_std * Hc, size = None)
        if y % 2 == 0:
            lattice[x, y, 3:6] = np.array([1., 0., 0.])
        else:
            lattice[x, y, 3:6] = np.array([0., 1., 0.])
        return lattice

    def get_unit_vector(self, x, y, negative = False, lattice = None):
        if lattice is None:
            lattice = self.lattice
        if y % 2 == 0:
            if negative:
                return np.array([-1., 0., 0.])
            else:
                return np.array([1., 0., 0.])
        else:
            if negative:
                return np.array([0., -1., 0.])
            else:
                return np.array([0., 1., 0.])

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
                               self.bar_length, self.bar_width, self.bar_thickness, self.magnetisation,
                               self.side_len_x, self.side_len_y, self.type, self.applied_field, self.field_angle,
                               self.Hc, self.Hc_std])
        np.savez_compressed(os.path.join(folder, file), self.lattice, parameters)

    def load(self, file):
        '''
        load in existing arrays
        '''
        if '.npz' not in file:
            file = file + '.npz'
        npzfile = np.load(file)
        parameters = npzfile['arr_1']
        self.unit_cells_x = np.int(parameters[0])
        self.unit_cells_y = np.int(parameters[1])
        self.vertex_gap = np.float(parameters[2])
        self.bar_length = np.float(parameters[3])
        self.bar_width = np.float(parameters[4])
        self.bar_thickness = np.float(parameters[5])
        self.magnetisation = np.float(parameters[6])
        self.side_len_x = np.int(parameters[7])
        self.side_len_y = np.int(parameters[8])
        self.type = parameters[9]
        self.applied_field = np.float(parameters[10])
        self.field_angle = np.float(parameters[11])
        # print(self.type)
        if len(parameters) > 12:
            self.Hc = np.float(parameters[12])
            self.Hc_std = np.float(parameters[13])
        self.lattice = npzfile['arr_0']

    # LATTICE TYPES
    def square(self, Hc_mean=0.03, Hc_std=0.05):
        '''
        Defines the lattice positions, magnetisation directions and coercive fields of an array of
        square ASI
        Takes the unit cell from the initial defined parameters
        Generates a normally distributed range of coercive fields of the bars using Hc_mean and Hc_std as a percentage
        One thing to potentially change is to have the positions in nanometers
        '''
        self.type = 'square'
        self.Hc = Hc_mean  # Unit cell direction in x andy y
        self.Hc_std = Hc_std
        self.side_len_x = 2 * self.unit_cells_x + 1
        self.side_len_y = 2 * self.unit_cells_y + 1
        bar_l, bar_w, bar_t = self.bar_length, self.bar_width, self.bar_thickness
        grid = np.zeros((self.side_len_x, self.side_len_y, 14))
        for x in range(0, self.side_len_x):
            for y in range(0, self.side_len_y):
                xpos = x * self.unit_cell_len
                ypos = y * self.unit_cell_len
                if (x + y) % 2 != 0:
                    Hc = np.random.normal(loc=Hc_mean, scale=Hc_std * Hc_mean, size=None)
                    if y % 2 == 0:
                        grid[x, y] = np.array([xpos, ypos, 0., 1., 0., 0., Hc, bar_l, bar_w, bar_t, None, 0, 0, 0])
                    else:
                        grid[x, y] = np.array([xpos, ypos, 0., 0., 1., 0., Hc, bar_l, bar_w, bar_t, None, 0, 0, 0])
                else:
                    if (x) % 2 == 0 and x != 0 and y != 0 and x != self.side_len_x - 1 and y != self.side_len_x - 1:
                        grid[x, y] = np.array([xpos, ypos, 0., 0., 0., 0., 0., None, None, None, 0, None, 0, 0])
                    else:
                        grid[x, y] = np.array([xpos, ypos, 0., 0., 0., 0., 0., None, None, None, None, None, 0, 0])
        self.lattice = grid

    def square_staircase(self, Hc_thin=0.03, Hc_thick=0.015, Hc_std=0.05, thick_bar_w=80e-9):
        self.square(Hc_thin, Hc_std)
        self.Hc_thick = Hc_thick
        lattice = copy.deepcopy(self.lattice)
        bar_w = lattice[:, :, 8]
        Hc = lattice[:, :, 6]
        for x in range(0, self.side_len_x):
            for y in range(0, self.side_len_y):
                Hc_new = np.random.normal(loc=Hc_thick, scale=Hc_std * Hc_thick, size=None)
                if x % 4 == 0 and y % 4 == 1:
                    bar_w[x, y] = thick_bar_w
                    Hc[x, y] = Hc_new
                    #lattice = self.set_vortex(x, y, lattice)
                elif x % 4 == 2 and y % 4 == 3:
                    bar_w[x, y] = thick_bar_w
                    Hc[x, y] = Hc_new
                    #lattice = self.set_vortex(x, y, lattice)
                elif x % 4 == 1 and y % 4 == 2:
                    bar_w[x, y] = thick_bar_w
                    Hc[x, y] = Hc_new
                    #lattice = self.set_vortex(x, y, lattice)
                elif x % 4 == 3 and y % 4 == 0:
                    bar_w[x, y] = thick_bar_w
                    Hc[x, y] = Hc_new
                    #lattice = self.set_vortex(x, y, lattice)
        self.lattice = lattice
        self.lattice[:, :, 8] = bar_w
        self.lattice[:, :, 6] = Hc

    def square_staircase_vortex(self, Hc_thin=0.03, Hc_thick=0.015, Hc_std=0.05, thick_bar_w=80e-9):
        self.square(Hc_thin, Hc_std)
        lattice = copy.deepcopy(self.lattice)
        bar_w = lattice[:, :, 8]
        Hc = lattice[:, :, 6]
        for x in range(0, self.side_len_x):
            for y in range(0, self.side_len_y):
                Hc_new = np.random.normal(loc=Hc_thick, scale=Hc_std * Hc_thick, size=None)
                if x % 4 == 0 and y % 4 == 1:
                    bar_w[x, y] = thick_bar_w
                    Hc[x, y] = Hc_new
                    lattice = self.set_vortex(x, y, lattice)
                elif x % 4 == 2 and y % 4 == 3:
                    bar_w[x, y] = thick_bar_w
                    Hc[x, y] = Hc_new
                    lattice = self.set_vortex(x, y, lattice)
                elif x % 4 == 1 and y % 4 == 2:
                    bar_w[x, y] = thick_bar_w
                    Hc[x, y] = Hc_new
                    lattice = self.set_vortex(x, y, lattice)
                elif x % 4 == 3 and y % 4 == 0:
                    bar_w[x, y] = thick_bar_w
                    Hc[x, y] = Hc_new
                    lattice = self.set_vortex(x, y, lattice)
        self.lattice = lattice
        self.lattice[:, :, 8] = bar_w
        self.lattice[:, :, 6] = Hc

    # APPLIED FIELD TYPES
    def AdaptiveField(self, Hmax, steps):
        '''
        Return array of field steps ranging from minimum Coercive field to Hmax
        Applied fields go from positive to negative
        (2 * steps) field values in a period
        '''
        testLattice = copy.deepcopy(self.lattice)
        testLattice[testLattice[:, :, 6] == 0] = np.nan
        Hc_min = np.nanmin(testLattice[:, :, 6])
        field_steps = np.linspace(Hc_min, Hmax, steps)
        field_steps = np.append(field_steps, np.negative(field_steps))
        return field_steps

    def SineField(self, Hmax, steps):
        '''
        return a array of field sweep values from 0 to Hmax in the form of a sine wave
        (2 * steps) field values in a period
        '''
        field_steps = Hmax * np.sin(np.linspace(0, 2 * np.pi, 2 * (steps)))
        return field_steps

    def LinearField(self, Hmax, steps):
        '''
        Return a array of field steps linearly increasing field from Hmin to Hmax
        (steps) number of field values
        '''
        testLattice = copy.deepcopy(self.lattice)
        testLattice[testLattice[:, :, 6] == 0] = np.nan
        Hmin = np.nanmin(testLattice[:, :, 6])
        field_steps = np.linspace(Hmin-0.006, Hmax, steps)
        field_steps = np.negative(field_steps)
        return field_steps

    # SIMULATION EXECUTABLES
    def fieldSweep(self, fieldType, Hmax, steps, Htheta, n=10, loops=1, folder=None,
                   FMR = False, FMR_field = None):
        '''
        Sweeps through the lattice using the designated field type.
        Total number of steps for a full minor loop is (2 * step).
        Allowed FieldTypes are:
            Sine, Adaptive
        The function then performs loops number of minor loops
        The Lattice after each field step gets saved to a folder. if folder is None then the
        function saves the lattice to the current working directory
        '''
        # Determine which field type to sweep the lattice
        field_steps = {
            'Sine': self.SineField(Hmax, steps),
            'Adaptive': self.AdaptiveField(Hmax, steps),
            'Linear': self.LinearField(Hmax, steps)
        }.get(fieldType, Exception('Field sweep type not defined'))
        # Working out field angle and amend field steps
        self.field_angle = Htheta
        Hrad = np.deg2rad(Htheta)
        if np.sin(Hrad) == 0:
            angleFactor = np.cos(Hrad)
        else:
            angleFactor = np.sin(Hrad)
        field_steps = field_steps / angleFactor
        # Create statistical parameters
        q, mag, monopole, fieldloops, frequency, vortex_count, macrospin_count = ([] for i in range(7))
        counter = 0
        i = 0
        # Start the field sweep
        print('STARTING SIMULATION WITH {} MINOR LOOPS...'.format(loops))
        if FMR:
            #freq = self.FMR(FMR_field, Htheta, n)
            freq = self.FMR_HM(n)
            frequency.append(freq)
        self.relax(n=n)
        self.save('InitialASVI_Hmax{:.3f}_steps{}_Angle{:.0f}_n{}_Loops{}'.format(
                   Hmax, steps, Htheta, n, loops), folder=folder)
        while i <= loops:
            self.previous = copy.deepcopy(self)
            for field in tqdm(field_steps, desc='Simulation Loop {} Progress: '.format(i),
                              unit='step'):
                self.applied_field = field
                Happlied = field * np.array([np.cos(Hrad), np.sin(Hrad), 0.])
                self.relax(Happlied, n)
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
            if FMR:
                #freq = self.FMR(FMR_field, Htheta, n)
                freq = self.FMR_HM(n)
                frequency.append(freq)
            i += 1
        self.save('FinalASVI_Hmax{:.3f}_steps{}_Angle{:.0f}_n{}_Loops{}'.format(
                   Hmax, steps, Htheta, n, loops), folder=folder)
        # Saving statistical information
        file = 'ASVIStateInfo_Hmax{:.3f}_steps{}_Angle{:.0f}_n{}_Loops{}'.format(Hmax, steps, Htheta, n, loops)
        parameters = np.array([Hmax, steps, Htheta, n, loops, self.Hc, self.Hc_std])
        if folder == None:
            folder = os.getcwd()
        np.savez(os.path.join(folder, file), parameters, fieldloops, q, mag, monopole, vortex_count, macrospin_count,
                 frequency)
        print('SIMULATION COMPLETE!')

    def FMR_HM(self, n=5):
        grid = copy.deepcopy(self.lattice)
        Xpos, Ypos = np.where(grid[:, :, 6] != 0)
        positions = np.array(list(zip(Xpos, Ypos)))

        freq = []
        for pos in positions:
            x = pos[0]
            y = pos[1]
            if grid[x, y, 11] == True:
                type = 2
            elif self.get_bar_width(x, y) > 150e-9:
                type = 1
            else:
                type = 0
            B = 1000 * np.linalg.norm(self.Hlocal(x, y, n=n))   # convert to mT
            frequency = FMR_heatmap(type = type, field = B)
            freq.append(frequency)
        return np.array(freq)

    def FMR(self, field=-0.0012, theta=45, n=5):
        rad = np.deg2rad(theta)
        Happlied = field * np.array([np.cos(rad), np.sin(rad), 0.])
        grid = copy.deepcopy(self.lattice)
        Xpos, Ypos = np.where(grid[:, :, 6] != 0)
        positions = np.array(list(zip(Xpos, Ypos)))

        freq = []
        for pos in positions:
            x = pos[0]
            y = pos[1]
            unit_vector = grid[x, y, 3:6]
            B = np.dot(np.array(Happlied + self.Hlocal(x, y, n=n)), unit_vector)
            M = self.magnetisation
            frequency = (2/2*np.pi)*np.sqrt(B * (B + 4*np.pi*10e-7*M))
            freq.append(frequency)
        return np.array(freq)

    def FMRAnimation(self, folder, fps=10):
        """
        Produce an animation of the lattice with frequency of each bar as it goes through the field loop
        just provide the folder where the field sweeps are saved
        """
        print('STARTING TO MAKE ANIMATION...')
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='ASVI FMR Simulation', artist='Matplotlib',
                        comment='Artificial Spin Vortex Ice Simulation')
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

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
                    freq = self.FMR_HM()
                    grid = copy.deepcopy(self.lattice)
                    # plotting vector field
                    ax = plot_vector_field_2D(grid, fig, ax, color = freq)
                    # setting title
                    H_applied = np.round(1000 * self.applied_field, 2)
                    H_theta = self.field_angle
                    steps = file[file.find('counter') + 7:file.find(r'_Loop')]
                    ax.set_title("Steps: " + steps, loc='left', pad=20)
                    ax.set_title('Applied Field: {} mT, Field Angle = {} deg'.format(H_applied, H_theta),
                                 loc='right', pad=20)
                    writer.grab_frame()
        print('ANIMATION COMPLETE!')

    def fieldSweepAnimation(self, folder, fps = 10):
        '''
        Will produce an animation of the lattice as it goes through the field sweep
        just provide the folder where the field sweeps are saved
        '''
        print('STARTING TO MAKE ANIMATION...')
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='ASVI Simulation', artist='Matplotlib',
                        comment='Artificial Spin Vortex Ice Simulation')
        writer = FFMpegWriter(fps = fps, metadata = metadata)
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

        def sortFunc(element):
            begin = element.find('counter') + 7
            end = element.find('_Loop')
            return (int(element[begin:end]))

        with writer.saving(fig, (os.path.join(folder, "animation.mp4")), 100):
            for root, dirs, files in os.walk(folder):
                new_files = list(filter(lambda x: 'ASVIcounter' in x, files))
                new_files.sort(key=sortFunc)
                print(new_files)
                for file in tqdm(new_files, desc='Animation Progress: ', unit='frame'):
                    ax.clear()
                    self.clearLattice()
                    self.load(os.path.join(root, file))
                    self.vertexCharge()
                    grid = copy.deepcopy(self.lattice)
                    # plotting vector field
                    ax = plot_vector_field_2D(grid, ax)
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
    def relax(self, Happlied=np.array([0., 0., 0.]), n=10):
        '''
        Steps through all the the positions in the lattice and if the field applied along the direction
        of the bar is negative and greater than the coercive field then it switches the magnetisation
        of the bar
        '''
        grid = copy.deepcopy(self.lattice)
        unrelaxed = True
        Happlied[Happlied == -0.] = 0.
        Xpos, Ypos = np.where(grid[:, :, 6] != 0)
        positions = np.array(list(zip(Xpos, Ypos)))

        while unrelaxed == True:
            flipcount = 0
            vortexcount = 0
            positions_new = np.random.permutation(positions)
            for pos in positions_new:
                x = pos[0]
                y = pos[1]
                if grid[x, y, 11] == 0:  # test if object is a macrospin
                    if abs(grid[x, y, 6]) != 0:
                        unit_vector = grid[x, y, 3:6]
                        field = np.dot(np.array(Happlied + self.Hlocal(x, y, n=n)), unit_vector)
                        if field < -grid[x, y, 6]:
                            if np.random.random() <= self.vortex_prob(x, y):
                                grid = self.set_vortex(x, y, grid)
                                grid[x, y, 13] += 1
                                vortexcount += 1
                            else:
                                grid[x, y, 3:6] = np.negative(grid[x, y, 3:6])
                                grid[x, y, 12] += 1
                                flipcount += 1
                elif grid[x, y, 11] == 1:  # test if object is a vortex
                    if abs(grid[x, y, 6]) != 0:
                        is_neg = Happlied<0
                        unit_vector = self.get_unit_vector(x, y, is_neg[0], grid)
                        field = np.dot(np.array(Happlied + self.Hlocal(x, y, n=n)), unit_vector)
                        if field < -grid[x, y, 6]:
                            grid = self.set_macrospin(x, y, grid)

            grid[grid == -0.] = 0.
            if flipcount > 0:
                unrelaxed = True
            else:
                unrelaxed = False
            self.lattice = grid

    def count_vortex(self):
        '''
        Count the number of vortices in the lattice
        '''
        types = self.lattice[:, :, 11]
        vortex_number = np.count_nonzero(types==1)
        return vortex_number

    def count_macrospin(self):
        '''
            Count the number of vortices in the lattice
        '''
        types = self.lattice[:, :, 11]
        macrospin_number = np.count_nonzero(types==0)
        return macrospin_number

    def vortex_prob(self, x, y, v_n=2):
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
        unique, count = np.unique(grid[x1:x2, y1:y2, 11], return_counts=True)

        min_width = 100e-9
        bar_width = self.get_bar_width(x, y)
        vortex_prob = 0
        if bar_width > min_width:  # thin bar below min_width cannot form vortex
            if self.applied_field < 0:
                if 1 in unique:
                    vortex_prob = 0.02 * count[1] + 0.0305  # slightly more likely for vortex to from beside vortices
                else:
                    vortex_prob = 0.0305
            else:
                if 1 in unique:
                    vortex_prob = 0.02 * count[1] + 0.0134
                else:
                    vortex_prob = 0.0134
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
                if l1[x, y, 6] != 0:
                    if np.array_equal(l1[x, y, 3:6], l2[x, y, 3:6]) == True:
                        same += 1.0
                    total += 1.0
        return (same / total)

    def netMagnetisation(self):
        '''
        returns the magnetisation in the x and y directions
        '''
        grid = copy.deepcopy(self.lattice)
        grid[grid[:, :, 6] == 0] = np.nan
        mx = grid[:, :, 3].flatten()
        my = grid[:, :, 4].flatten()
        return (np.array([np.nanmean(mx), np.nanmean(my)]))

    def monopoleDensity(self):
        '''
        Returns the monopole density of a square or kagome lattice
        '''
        # 4in/0out have a charge of 1
        # 3in/1out have a charge of 0.5
        #   The density is then calculated by dividing by the total area minus the edges
        self.vertexCharge()
        grid = self.lattice
        magcharge = grid[:, :, 10].flatten()
        return (np.nanmean(np.absolute(magcharge)))

    def Hlocal(self, x, y, n=1):
        if self.periodicBC:
            # calculates the local field at position x, y including the
            # field with n radius with n=1 just including nearest neighbours
            Hl = []
            x1 = x - n
            x2 = x + n + 1
            y1 = y - n
            y2 = y + n + 1
            indx = np.arange(x1, x2)
            indy = np.arange(y1, y2)
            # print(x1,x2,y1,y2)
            # indpara = np.arange(0,9)
            # indices = np.meshgrid(indx,indy, indpara)
            grid = np.take(self.lattice, indx, axis=0, mode='wrap')
            grid1 = np.take(grid, indy, axis=1, mode='wrap')
            # print(grid1[:,:,0:2])
            m = grid1[:, :, 3:6]
            m = m.reshape(-1, m.shape[-1])
            r = grid1[:, :, 0:3]
            r = r.reshape(-1, r.shape[-1])
            r0 = self.lattice[x, y, 0:3]
            # if abs(np.linalg.norm(pos-r0))/(n+1)>=1.0:
            #    pos = np.array([self.side_len_x*self.unit_cell_len, self.side_len_y*self.unit_cell_len, 0])+ pos
            # r[:,:,0][np.where(abs(np.linalg.norm(r[:,:,0]-r0[0]))/(n+1)>1.0:)] = self.side_len_x*self.unit_cell_len+ r
            for pos, mag in zip(r, m):
                # print(np.linalg.norm(pos-r0)/(self.unit_cell_len*(n+1)))
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
                        # if abs(np.linalg.norm(pos-r0))/(n+1)>1.0:
                #    pos = np.array([self.side_len_x*self.unit_cell_len, self.side_len_y*self.unit_cell_len, 0])+ pos
                if np.array_equal(pos, r0) != True:  # abs(np.linalg.norm(pos-r0))/(n+1)<=1.0
                    Hl.append(self.fieldCalc(mag, r0, pos))
            return (sum(Hl))
        else:
            # calculates the local field at position x, y including the
            # field with n radius with n=1 just including nearest neighbours
            Hl = []
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

            grid = self.lattice[x1:x2, y1:y2, :]
            m = grid[:, :, 3:6]
            m = m.reshape(-1, m.shape[-1])
            r = grid[:, :, 0:3]
            r = r.reshape(-1, r.shape[-1])
            r0 = self.lattice[x, y, 0:3]

            for pos, mag in zip(r, m):
                if np.linalg.norm(pos - r0) / (n + 1) <= 1.0 and np.array_equal(pos, r0) != True:
                    Hl.append(self.fieldCalc(x, y, mag, r0, pos))
            return sum(Hl)

    def fieldCalc(self, x, y, mag, r0, pos):
        '''
        Tells the class what type of field calculation method
        you want to use for the rest of the simulation
        '''
        bar_length = self.get_bar_length(x, y)
        bar_width = self.get_bar_width(x, y)
        bar_thickness = self.get_bar_thickness(x, y)
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
            mag_charge = bar_thickness * self.magnetisation * bar_width
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
        for x in np.arange(0, self.side_len_x):
            for y in np.arange(0, self.side_len_y):
                if np.isnan(grid[x, y, 10]) != True:
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
                    charge = -(np.sum(local[0:2, 0:2, 3:6]) - np.sum(local[1:3, 1:3, 3:6])) / np.count_nonzero(
                        local[:, :, 6])

                    if self.type == 'kagome':
                        if x == 0:
                            charge = np.sum(local[0, :, 3]) - np.sum(local[1, :, 3]) + np.sum(local[:, 0, 4]) - np.sum(
                                local[:, 2, 4])
                        elif x == self.side_len_x - 1:
                            charge = np.sum(local[:, :, 3]) - np.sum(local[:, :, 3]) + np.sum(local[:, 0, 4]) - np.sum(
                                local[:, 2, 4])
                        elif (x - 2) % 4:
                            if (y - 2) % 4:
                                charge = np.sum(local[0, :, 3]) - np.sum(local[1, :, 3]) + np.sum(
                                    local[:, 0, 4]) - np.sum(local[:, 2, 4])
                            else:
                                charge = np.sum(local[1, :, 3]) - np.sum(local[2, :, 3]) + np.sum(
                                    local[:, 0, 4]) - np.sum(local[:, 2, 4])
                        elif (x) % 4:
                            if (y - 2) % 4:
                                charge = np.sum(local[1, :, 3]) - np.sum(local[2, :, 3]) + np.sum(
                                    local[:, 0, 4]) - np.sum(local[:, 2, 4])
                            else:
                                charge = np.sum(local[0, :, 3]) - np.sum(local[1, :, 3]) + np.sum(
                                    local[:, 0, 4]) - np.sum(local[:, 2, 4])
                        if charge > 3:
                            charge = 1
                        elif charge < -3:
                            charge = -1
                        else:
                            charge = 0

                    grid[x, y, 10] = charge

        self.lattice = grid

    def determinePeriod(self, folder, Hmax='1p414214e-01_Angle'):
        '''
        Determines the period in the minor loop.
        '''
        print(Hmax)
        print(type(Hmax))
        filenames_pos = []
        filenames_neg = []
        latticelist_pos = []
        latticelist_neg = []
        for root, sub, files in os.walk(folder):
            for file in files:
                checkstr = 'd' + Hmax
                print(str(checkstr), file, str(checkstr) in file, file.find(checkstr))
                if checkstr in file:
                    print('pos')
                    filename = os.path.join(root, file)
                    filenames_pos.append(filename)
        # filenames_pos.sort(key = self.sortFunc2)
        filenames_pos.sort(key=lambda s: os.path.getmtime(s))
        # print(filenames_pos)
        for root, sub, files in os.walk(folder):
            for file in files:
                checkstr = '-' + Hmax
                print(checkstr, file, checkstr in file)
                if checkstr in file:
                    print('neg')
                    filename = os.path.join(root, file)
                    filenames_neg.append(filename)
        # filenames_neg.sort(key = self.sortFunc2)
        filenames_neg.sort(key=lambda s: os.path.getmtime(s))
        print(filenames_pos, filenames_neg)
        for i, file_pos, file_neg in zip(np.arange(0, len(filenames_neg)), reversed(filenames_pos),
                                         reversed(filenames_neg)):
            if i == 0:
                self.load(file_pos)
                latticelist_pos.append(self.returnLattice())
                self.load(file_neg)
                latticelist_neg.append(self.returnLattice())
                continue
            self.load(file_pos)
            latticelist_pos.append(self.returnLattice())
            self.load(file_neg)
            latticelist_neg.append(self.returnLattice())
            # self.graphCharge()
            # print(i, len(latticelist_neg))
            corr_pos = np.array_equal(latticelist_pos[0][:, :, 3:6], latticelist_pos[i][:, :, 3:6])
            corr_neg = np.array_equal(latticelist_neg[0][:, :, 3:6], latticelist_neg[i][:, :, 3:6])
            print(corr_pos, corr_neg, i)
            if corr_pos == True and corr_neg == True:
                return (i)
                break
        self.load(filenames_neg[-1])

