import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy.linalg
from tqdm import tqdm
# Vortex Spin Modules
from plotting import plot_vector_field_2D, FMR_heatmap
from asvi_materials import NanoBar, Vertex

plt.rcParams['animation.ffmpeg_path'] = r'E:\ASI_MSci_Project\ffmpeg\bin\ffmpeg.exe'


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
    def __init__(self, unit_cells_x=25, unit_cells_y=25, vertex_gap=100e-9,
                 interType='dumbbell', periodicBC=False):
        # Lattice Parameters
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
    def square(self, hc_m=0.03, hc_v=0.02, hc_std=0.05, magnetisation=800e3,
               bar_l=600e-9, bar_w=125e-9, bar_t=20e-9):
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
                        grid[x].append(NanoBar(xpos, ypos, 0., 1., 0., 0., hc_m, hc_v, hc_std,
                                               bar_l, bar_w, bar_t, magnetisation, 'macrospin'))
                    else:
                        grid[x].append(NanoBar(xpos, ypos, 0., 0., 1., 0., hc_m, hc_v, hc_std,
                                               bar_l, bar_w, bar_t, magnetisation, 'macrospin'))
                else:
                    if (x) % 2 == 0 and x != 0 and y != 0 and x != self.side_len_x - 1 and y != self.side_len_x - 1:
                        grid[x].append(Vertex(xpos, ypos, 0.))
                    else:
                        grid[x].append(Vertex(xpos, ypos, None))
        self.lattice = np.asarray(grid, dtype=object)

    def square_staircase(self, hc_thin=0.03, hc_thick=0.015, hc_v=0.02, hc_std=0.05, magnetisation=800e3,
                         bar_l=600e-9, thin_bar_w=125e-9, thick_bar_w=200e-9, bar_t=20e-9):
        self.square(hc_thin, hc_v, hc_std, magnetisation, bar_l, thin_bar_w, bar_t)
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

    # APPLIED FIELD TYPES
    def AdaptiveField(self, Hmax, steps):
        '''
        Return array of field steps ranging from minimum Coercive field to Hmax
        Applied fields go from positive to negative
        (2 * steps) field values in a period
        '''
        hc_min = np.nanmin(self.hc_matrix())
        field_steps = np.linspace(hc_min, Hmax, steps)
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
        hc_min = np.nanmin(self.hc_matrix())
        field_steps = np.linspace(hc_min - 0.006, Hmax, steps)
        field_steps = np.negative(field_steps)
        return field_steps

    # SIMULATION EXECUTABLES
    def fieldSweep(self, fieldType, Hmax, steps, Htheta, n=10, loops=1, folder=None,
                   FMR=False, FMR_field=None):
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
                # freq = self.FMR(FMR_field, Htheta, n)
                freq = self.FMR_HM(n)
                frequency.append(freq)
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

    def FMR_HM(self, n=5):
        grid = copy.deepcopy(self.lattice)
        Xpos, Ypos = np.nonzero(grid)
        positions = np.array(list(zip(Xpos, Ypos)))

        freq = []
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
                B = 1000 * np.linalg.norm(self.Hlocal(x, y, n=n))  # convert to mT
                frequency = FMR_heatmap(type=tp, field=B, bias=obj.hc_bias)
                freq.append(frequency)
        return np.array(freq)

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
    def relax(self, Happlied=np.array([0., 0., 0.]), n=10):
        '''
        Steps through all the the positions in the lattice and if the field applied along the direction
        of the bar is negative and greater than the coercive field then it switches the magnetisation
        of the bar
        '''
        grid = copy.deepcopy(self.lattice)
        unrelaxed = True
        Xpos, Ypos = np.nonzero(grid)
        positions = np.array(list(zip(Xpos, Ypos)))

        while unrelaxed == True:
            flipcount = 0
            vortexcount = 0
            positions_new = np.random.permutation(positions)
            for pos in positions_new:
                x = pos[0]
                y = pos[1]
                obj = grid[x, y]
                if obj.type == 'macrospin':  # test if object is a macrospin
                    unit_vector = obj.unit_vector
                    field = np.dot(np.array(Happlied + self.Hlocal(x, y, n=n)), unit_vector)
                    if field < -obj.hc:
                        obj.flip()
                        flipcount += 1
                        if np.random.random() <= self.vortex_prob(x, y):
                            obj.set_vortex()
                            vortexcount += 1
                elif obj.type == 'vortex':  # test if object is a vortex
                    unit_vector = obj.unit_vector
                    field = np.dot(np.array(Happlied + self.Hlocal(x, y, n=n)), unit_vector)
                    if field < -obj.hc:
                        obj.set_macrospin()
                        obj.flip()

            if flipcount > 0:
                unrelaxed = True
            else:
                unrelaxed = False
            self.lattice = grid

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

            grid = self.lattice
            obj = self.lattice[x, y]
            r0 = obj.pos
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
                if type(obj) == Vertex:
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


