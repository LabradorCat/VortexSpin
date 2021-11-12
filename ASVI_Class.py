import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import scipy.interpolate as spi
import matplotlib.animation as manimation
plt.rcParams['animation.ffmpeg_path'] = r'E:\ASI_MSci_Project\ffmpeg-2021-11-07-git-45dc668aea-full_build\bin\ffmpeg.exe'


class ASVI():
    '''
    Class for initialising the lattice be performing field sweeps, mainly
    return point memory
    '''
    def __init__(self, unit_cells_x=25, unit_cells_y=25, lattice = None, \
        bar_length = 220e-9, vertex_gap = 1e-7, bar_thickness = 25e-9, \
        bar_width = 80e-9, magnetisation = 800e3):
        self.lattice = lattice
        self.type = None
        self.Hc = None
        self.Hc_std = None
        self.previous = None
        self.unit_cells_x = unit_cells_x
        self.unit_cells_y = unit_cells_y
        self.side_len_x = None      #The side length is now defined in the square lattice
        self.side_len_y = None
        self.bar_length = bar_length
        self.vertex_gap = vertex_gap
        self.bar_width = bar_width
        self.bar_thickness = bar_thickness
        self.width = bar_width
        self.magnetisation = magnetisation
        self.periodicBC = False
        self.unit_cell_len = (bar_length+vertex_gap)/2
        self.interType = 'dumbbell'

    def returnLattice(self):
        '''
        Returns the lattice in its current state
        '''
        return(self.lattice)

    def clearLattice(self):
        '''
        Clears the lattice
        '''
        self.lattice = None

    def save(self, file, folder = os.getcwd()):
        '''
        Save existing arrays
        '''
        if folder == None:
            folder = os.getcwd()
        file = file.replace('.','p')
        parameters = np.array([self.unit_cells_x,self.unit_cells_y,\
            self.bar_length,self.vertex_gap,self.bar_width,\
            self.bar_thickness,self.magnetisation, self.side_len_x, self.side_len_y, self.type,\
            self.Hc, self.Hc_std])
        np.savez_compressed(os.path.join(folder,file), self.lattice, parameters)

    def load(self, file):
        '''
        load in existing arrays
        '''
        if '.npz' not in file:
            file = file+'.npz'
        npzfile = np.load(file)
        #print(npzfile)
        #print(npzfile.files)
        #print(npzfile.f.arr_1)
        parameters = npzfile['arr_1']
        #print(len(parameters))
        self.unit_cells_x = np.int(parameters[0])
        self.unit_cells_y = np.int(parameters[1])
        self.bar_length = np.float(parameters[2])
        self.vertex_gap = np.float(parameters[3])
        self.bar_width = np.float(parameters[4])
        self.bar_thickness = np.float(parameters[5])
        self.magnetisation = np.float(parameters[6])
        self.side_len_x = np.int(parameters[7])
        self.side_len_y = np.int(parameters[8])
        self.type = parameters[9]
        #print(self.type)
        if len(parameters) > 10:
            self.Hc = np.float(parameters[10])
            self.Hc_std = np.float(parameters[11])
        self.lattice = npzfile['arr_0']

    '''
       These are the functions that define the lattice type and
       position of each of the bars:
           - Square
           - Tilted square
           - Kagome
           - Short shakti
           - Long shakti
           - Tetris
       The lattice is stored as a numpy array [x_position, y_position, z_position,
       x_magnetisation, y_magnetisation, z_magnetisation, coercive field,
       flip_count, vertex or not]
       '''
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
        grid = np.zeros((2 * self.unit_cells_x + 1, 2 * self.unit_cells_y + 1, 9))
        for x in range(0, 2 * self.unit_cells_x + 1):
            for y in range(0, 2 * self.unit_cells_y + 1):
                if (x + y) % 2 != 0:
                    if y % 2 == 0:
                        xpos = x * (self.bar_length + self.vertex_gap) / 2
                        ypos = y * (self.bar_length + self.vertex_gap) / 2
                        grid[x, y] = np.array([x * self.unit_cell_len, y * self.unit_cell_len, 0., 1., 0., 0.,
                                               np.random.normal(loc=Hc_mean, scale=Hc_std * Hc_mean, size=None), 0,
                                               None])
                    else:
                        grid[x, y] = np.array([x * self.unit_cell_len, y * self.unit_cell_len, 0., 0., 1., 0.,
                                               np.random.normal(loc=Hc_mean, scale=Hc_std * Hc_mean, size=None), 0,
                                               None])
                else:
                    if (x) % 2 == 0 and x != 0 and y != 0 and x != self.side_len_x - 1 and y != self.side_len_x - 1:
                        # print('test')
                        grid[x, y] = np.array([x * self.unit_cell_len, y * self.unit_cell_len, 0., 0., 0., 0., 0, 0, 0])
                    else:
                        grid[x, y] = np.array(
                            [x * self.unit_cell_len, y * self.unit_cell_len, 0., 0., 0., 0., 0, 0, None])
        print(grid)
        self.lattice = grid

    '''
        These are simulation executables
    '''
    def fieldSweepAdaptive(self, Hmax, steps, Htheta, n=10, loops=1, folder=None, q1=False):
        '''
        Sweeps through from 90% of the minimum Coercive field to Hmax at angle Htheta in steps.
        Total number of steps for a full minor loop is 4*(step+1).
        The function then performs loops number of minor loops
        The Lattice after each field step gets saved to a folder. if folder is None then the
        function saves the lattice to the current working directory
        '''
        if folder == None:
            folder = os.getcwd()
        M0 = copy.deepcopy(self)
        testLattice = copy.deepcopy(self.lattice)
        Htheta = np.pi * Htheta / 180
        testLattice[testLattice[:, :, 6] == 0] = np.nan
        if np.sin(Htheta) == 0:
            angleFactor = np.cos(Htheta)
        else:
            angleFactor = np.sin(Htheta)
        Hc_min = np.nanmin(testLattice[:, :, 6]) / angleFactor
        Hc_array = testLattice[:, :, 6].flatten()
        Hc_array = np.append(Hc_array, Hmax)
        Hc_array.sort()

        Hc_func = spi.interp1d(np.linspace(0, 1, Hc_array.size), Hc_array)
        Hc_new = Hc_func(np.linspace(0, 1, 1000))

        field_steps = Hc_new[np.where(Hc_new <= Hmax)]
        idx = np.round(np.linspace(0, len(field_steps) - 1, steps)).astype(int)
        field_steps = field_steps[idx]
        field_steps = np.append(field_steps, Hmax)
        # plt.figure()
        # print(Hc_array.size, Hc_new.size, field_steps.size)

        # plt.plot(np.linspace(0,4, Hc_array.size),Hc_array,'o-')
        # plt.plot(np.linspace(0,4,1000),Hc_new)
        # plt.plot(np.linspace(0,1, field_steps.size),field_steps, '.')
        # plt.show()
        field_neg = -1 * field_steps
        field_steps = np.append(field_steps, field_neg)
        idx = np.append(idx, idx[-1] + 1)
        print(Hc_array)
        field_steps = field_steps / angleFactor
        q = []
        mag = []
        monopole = []
        fieldloops = []
        vertex = []
        counter = 0
        period = None
        i = 0
        self.relax(n=n)
        tcycles = 15
        if folder == None:
            self.save(
                'InitialRPMLattice_Hmax%(Hmax)e_steps%(steps)d_Angle%(Htheta)e_neighbours%(n)d_Loops%(loops)d' % locals(),
                folder=folder)
        else:
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.save(
                'InitialRPMLattice_Hmax%(Hmax)e_steps%(steps)d_Angle%(Htheta)e_neighbours%(n)d_Loops%(loops)d' % locals(),
                folder=folder)
        while i <= loops:
            self.previous = copy.deepcopy(self)
            for field in field_steps:
                Happlied = field * np.array([np.cos(Htheta), np.sin(Htheta), 0.])
                print('Happlied: ', Happlied)
                print()
                self.relax(Happlied, n)
                fieldloops.append(np.array([i, field]))
                mag.append(self.netMagnetisation())
                monopole.append(self.monopoleDensity())
                q.append(self.correlation(self.previous, self))
                # vertex.append(self.vertexTypePercentage())
                if not os.path.exists(folder):
                    os.makedirs(folder)
                self.save('Lattice_counter%(counter)d_Loop%(i)d_FieldApplied%(field)e_Angle%(Htheta)e' % locals(),
                          folder=folder)
                counter += 1

            if q1 == True and period == None:
                finalfield = abs(field)
                namestr = '%(finalfield)e_A'% locals()
                print(namestr)
                period = self.determinePeriod2(folder, Hmax = namestr.replace('.', 'p'))
                print('period:', period)
                if period != None:
                    loops = i + period
                    tcycles=i
            i += 1
            print(i,loops, period)

        self.save('FinalRPMLattice_Hmax%(Hmax)e_steps%(steps)d_Angle%(Htheta)e_neighbours%(n)d_Loops%(loops)d' % locals(),folder=folder)
        fieldloops = np.array(fieldloops)
        q = np.array(q)
        mag = np.array(mag)
        monopole = np.array(monopole)
        vertex = np.array(vertex)
        file = 'RPMStateInfo_Hmax%(Hmax)e_steps%(steps)d_Angle%(Htheta)e_neighbours%(n)d_Loops%(loops)d' % locals()
        parameters = np.array([Hmax, steps, Htheta, n, loops, self.Hc, self.Hc_std, period, tcycles])
        print(parameters)
        if folder == None:
            folder = os.getcwd()
            np.savez(os.path.join(folder, file), parameters, fieldloops, q, mag, monopole, vertex)
        else:
            np.savez(os.path.join(folder, file), parameters, fieldloops, q, mag, monopole, vertex)

    def relax(self, Happlied = np.array([0.,0.,0.]), n=10):
        '''
        Steps through all the the positions in the lattice and if the field applied along the direction
        of the bar is negative and greater than the coercive field then it switches the magnetisation
        of the bar
        '''
        grid = copy.deepcopy(self.lattice)
        unrelaxed = True
        Happlied[Happlied == -0.] = 0.
        #Xpos = np.random.permutation(grid[:,:, 0].flatten()).tolist()
        #Ypos = np.random.permutation(grid[:,:, 1].flatten()).tolist()
        #print(Xpos, r'\n', Ypos)
        Xpos, Ypos = np.where(grid[:,:,6] != 0)
        positions = np.array(list(zip(Xpos, Ypos)))
        #print(positions)
        #Xpos = np.random.permutation(Xpos)
        #Ypos = np.random.permutation(Ypos)
        #print(Xpos, Ypos.tolist())
        #positions = np.random.permutation(positions)
        #print(positions)
        #time.sleep(20)
        total_flipcount = 0
        while unrelaxed == True:
            flipcount = 0
            vortexcount = 0
            positions_new = np.random.permutation(positions)
            for pos in positions_new:
                #print(pos, pos[0], pos[1])
                x = pos[0]
                y = pos[1]
                if abs(grid[x,y,6]) != 0:
                    unit_vector = grid[x,y,3:6]
                    field = np.dot(np.array(Happlied+self.Hlocal(x,y, n=n)), unit_vector)
                    #print(field)
                    if field < -grid[x,y,6]:
                        vortex_prob = np.random.randint(0,100)
                        if vortex_prob <= 10:
                            grid[x,y,3:9] = 0
                            vortexcount += 1
                            print('Vortex loc', x, y)
                        else:
                            #print(grid[x,y,3:5])
                            grid[x,y,3:5] = np.negative(grid[x,y,3:5])
                        #print(grid[x,y,3:5])
                            grid[x,y,:][grid[x,y,:] == 0.] = 0.
                            grid[x,y,7] += 1
                            flipcount += 1
                            #print(grid[x,y,3:5])
                            print('Flip loc', x, y)
                        #positions_new = np.append(positions_new, [[x+2, y],[x, y+2],[x-2, y],[x, y-2]])
                else:
                    print('wrong spin')
            print("no of flipped spins in relax", flipcount)
            grid[grid == -0.] = 0.
            if flipcount > 0:
                unrelaxed = True
            else:
                unrelaxed = False
            self.lattice = grid
            total_flipcount+=flipcount
        print(total_flipcount)

    def fieldSweepAnimation2(self, folder, name='Lattice_counter'):
        '''
        Will produce an animation of the lattice as it goes through the field sweep
        just provide the folder where the field sweeps are saved
        '''
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='a red circle following a blue sine wave')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        fig, ax = plt.subplots()
        # n = len(df['number'])
        ims = []
        counter = []
        fig_anim = plt.figure('Animation')

        def sortFunc(element):
            # print(element)
            begin = element.find('counter') + 7
            end = element.find('_Loop')
            # print(element.find('counter'))
            # print(element.find('_Loop'))
            # print(int(element[begin:end]))
            return (int(element[begin:end]))

        with writer.saving(fig, (os.path.join(folder, "animation.mp4")), 100):
            for root, dirs, files in os.walk(folder):
                new_files = list(filter(lambda x: 'Lattice_counter' in x, files))
                new_files.sort(key=sortFunc)
                for file in new_files:
                    # print(file)

                    # plt.clf()
                    # fig.clear()
                    ax.clear()
                    self.clearLattice()
                    self.load(os.path.join(root, file))
                    self.vertexCharge()
                    grid = self.lattice
                    X = grid[:, :, 0].flatten()
                    Y = grid[:, :, 1].flatten()
                    z = grid[:, :, 2].flatten()
                    Mx = grid[:, :, 3].flatten()
                    My = grid[:, :, 4].flatten()
                    Mz = grid[:, :, 5].flatten()
                    Hc = grid[:, :, 6].flatten()
                    C = grid[:, :, 7].flatten()
                    MagCharge = grid[:, :, 8].flatten()
                    # print(MagCharge)

                    # fig = plt.figure(figsize=(6,6), num = 'test')
                    # ax = fig.add_subplot(111)
                    ax.set_xlim([-1 * self.unit_cell_len, np.max(X) + self.unit_cell_len])
                    ax.set_ylim([-1 * self.unit_cell_len, np.max(Y) + self.unit_cell_len])
                    # ax.set_title("Vertex Magnetic Charge Map",fontsize=14)
                    # ax.set_xlabel("XAVG",fontsize=12)
                    # ax.set_ylabel("YAVG",fontsize=12)
                    # ax.grid(True,linestyle='-',color='0.75')

                    ax.quiver(X, Y, Mx, My, angles='xy', scale_units='xy', pivot='mid', zorder=1)
                    # quiver.set_clim(self, 0, 2)
                    # scatter with colormap mapping to z value
                    ax.scatter(X, Y, s=50, c=MagCharge, cmap='gist_rainbow', marker='o', zorder=2, vmax=1, vmin=-1)

                    # cb2 = fig.colorbar(graph, fraction=0.046, pad=0.04, ax = ax)
                    # ax.set(adjustable='box', aspect='equal')
                    plt.ticklabel_format(style='sci', scilimits=(0, 0))
                    plt.tight_layout()

                    ax.set_title(file[file.find('counter') + 7:file.find(r'_Loop')])
                    # print(file.find('counter'),file.find(r'_Loop'))
                    # print(counter)
                    # plt.show()
                    writer.grab_frame()
                    # ims.append([im])
        # sorted_ims = [x for _,x in sorted(zip(counter,ims))]
        # anim = pla.ArtistAnimation(fig_anim, ims, interval = 100, blit = True, repeat_delay = 1000)
        # Writer = pla.writers['ffmpeg']
        # writer = pla.FFMpegWriter(fps=15, metadata=dict(artist='Alex Vanstone'), bitrate=1800)
        # anim.save(os.path.join(folder, 'Video.mp4'), writer = writer)
        # plt.show()

    '''
        Basic Calculations
    '''
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
                if l1[x,y,6]!=0:
                    if np.array_equal(l1[x,y, 3:6], l2[x,y,3:6]) ==True:
                        same+=1.0
                    total +=1.0
        #print("Same total:",same)
        #print("Absolute total:", total)
        #print('Correlation factor:',same/total)
        return(same/total)

    def netMagnetisation(self):
        '''
        returns the magnetisation in the x and y directions
        '''
        grid = copy.deepcopy(self.lattice)
        grid[grid[:,:,6]==0] = np.nan
        mx = grid[:,:,3].flatten()
        my = grid[:,:,4].flatten()
        return(np.array([np.nanmean(mx),np.nanmean(my)]))

    def monopoleDensity(self):
        '''
        Returns the monopole density of a square or kagome lattice
        '''
        #4in/0out have a charge of 1
        #3in/1out have a charge of 0.5
        #The density is then calculated by dividing by the total area minus the edges
        self.vertexCharge()
        grid = self.lattice
        magcharge = grid[:,:,8].flatten()
        return(np.nanmean(np.absolute(magcharge)))

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
                    Hl.append(self.fieldCalc(mag, r0, pos))
            return (sum(Hl))

    def vertexCharge(self):
        '''
        Works you the vertex charge for square and Kagome.
        Should work on tetris and shakti but haven't test it yet
        '''
        grid = copy.deepcopy(self.lattice)
        for x in np.arange(0, self.side_len_x):
            for y in np.arange(0, self.side_len_y):
                if np.isnan(grid[x, y, 8]) != True:
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

                    grid[x, y, 8] = charge

        self.lattice = grid

    def fieldCalc(self, mag, r0, pos):
        '''
        Tells the class what type of field calculation method
        you want to use for the rest of the simulation
        '''
        if self.interType == 'dipole':
            # Calculate a field in point r created by a dipole moment m located in r0.
            # Spatial components are the outermost axis of r and returned B.
            mag = np.array(mag)
            r0 = np.array(r0)
            pos = np.array(pos)
            mag = self.magnetisation * self.bar_length * self.bar_width * self.bar_thickness * mag
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
            return (B)
        if self.interType == 'dumbbell':
            # Using the dumbbell model to calculate the interaction between each bar
            mag = np.array(mag)
            r0 = np.array(r0)
            pos = np.array(pos)
            mag_charge = self.bar_thickness * self.magnetisation * self.bar_width
            r2 = np.subtract(np.transpose(r0), pos).T + mag * self.bar_length / 2
            r1 = np.subtract(np.transpose(r0), pos).T - mag * self.bar_length / 2
            B = 1e-7 * mag_charge * (r1 / np.linalg.norm(r1) ** 3 - r2 / np.linalg.norm(r2) ** 3)
            return (B)

    def determinePeriod2(self, folder, Hmax = '1p414214e-01_Angle'):
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
                checkstr = 'd'+Hmax
                print(str(checkstr), file, str(checkstr) in file, file.find(checkstr))
                if checkstr in file:
                    print('pos')
                    filename = os.path.join(root,file)
                    filenames_pos.append(filename)
        #filenames_pos.sort(key = self.sortFunc2)
        filenames_pos.sort(key=lambda s: os.path.getmtime(s))
        #print(filenames_pos)
        for root, sub, files in os.walk(folder):
            for file in files:
                checkstr = '-'+Hmax
                print(checkstr, file, checkstr in file)
                if checkstr in file:
                    print('neg')
                    filename = os.path.join(root,file)
                    filenames_neg.append(filename)
        #filenames_neg.sort(key = self.sortFunc2)
        filenames_neg.sort(key=lambda s: os.path.getmtime(s))
        print(filenames_pos, filenames_neg)
        for i, file_pos, file_neg in zip(np.arange(0,len(filenames_neg)), reverse(filenames_pos), reverse(filenames_neg)):
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
            #self.graphCharge()
            #print(i, len(latticelist_neg))
            corr_pos = np.array_equal(latticelist_pos[0][:,:,3:6], latticelist_pos[i][:,:,3:6])
            corr_neg = np.array_equal(latticelist_neg[0][:,:,3:6], latticelist_neg[i][:,:,3:6])
            print(corr_pos, corr_neg, i)
            if corr_pos == True and corr_neg == True:
                return(i)
                break
        self.load(filenames_neg[-1])




