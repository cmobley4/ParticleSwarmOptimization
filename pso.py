from math import *
import numpy as np


class PSO:
    def __init__(self, obj_func, N=20, world_bounds=((-50, 50), (-50, 50)), inertia=1.0, cog=2.0, soc=2.0, max_vel=2.0):
        # Initialize objective function
        self.obj_func = obj_func

        # Initialize scalar components
        self.inertia = inertia
        self.cognition = cog
        self.social = soc
        self.max_velocity = max_vel

        # Initialize world bounds
        self.world_bounds = np.array(world_bounds)

        # Generate particles and set initial personal best positions
        self.position = np.array([np.random.uniform(bounds[0], bounds[1], N) for bounds in world_bounds])
        self.velocity = np.zeros(self.position.shape)
        self.pbest = np.copy(self.position)

        # Find initial global best position
        self.gbest = self.position[:, 0].flatten()
        self.update_bests()

    # Return particle positions for given dimension
    def get_dim_pos(self, dim):
        return self.position[:, dim]

    # Update personal/global best based on current particle positions
    def update_bests(self):
        # apply fitness function
        fit = np.apply_along_axis(self.obj_func, 0, self.position, self.world_bounds[:, 1])
        pbest_fit = np.apply_along_axis(self.obj_func, 0, self.pbest, self.world_bounds[:, 1])

        # update personal bests
        for i, (f, pb) in enumerate(zip(fit, pbest_fit)):
            self.pbest[:, i] = self.position[:, i] if f > pb else self.pbest[:, i]
        
        # update global best
        max_ind = np.argmax(fit)
        gbest_fit = self.obj_func(self.gbest, self.world_bounds[:, 1])
        self.gbest = self.position[:, max_ind] if fit[max_ind] > gbest_fit else self.gbest

    # Step population by num_step steps
    def step(self):
        # Generate two random numbers
        rand_1 = np.random.random(self.position.shape[1])
        rand_2 = np.random.random(self.position.shape[1])
        
        # Determine particle velocities for this step
        self.velocity = self.inertia * self.velocity +\
            self.cognition * rand_1 * (self.pbest - self.position) +\
            self.social * rand_2 * (self.gbest.reshape(self.gbest.size, 1) - self.position)
        
        # correct particle velocities if they are beyond the max allowed
        v_sum_sqr = np.sum(np.power(self.velocity, 2), 0)
        for i, v in enumerate(v_sum_sqr):
            if v > pow(self.max_velocity, 2):
                self.velocity[:, i] = (self.max_velocity / sqrt(v)) * self.velocity[:, i]

        # Update particle positions
        self.position = self.position + self.velocity

        self.update_bests()

    # Get vector distances from global best for all particles
    def get_dists(self):
        return self.position - self.gbest.reshape(self.gbest.size, 1)

    # Get average vector distance to global best
    def get_avg_dist(self):
        return abs(np.average(self.position, axis=1) - self.gbest)

    # Get the number of particles which have converged within epsilon distance from global best
    def get_num_converged(self, epsilon=0.01):
        return np.count_nonzero(np.all(self.get_dists() < epsilon, axis=0))

    # Get vectored root mean square deviation from global best
    def get_rmsd(self):
        return np.sqrt(np.sum(np.power(self.get_dists(), 2), axis=1) / (2 * self.position.shape[1]))

    # Step system until rmsd is within threshold for all dimensions or until max_epochs is reached
    # Return the number of epochs before convergence
    def run(self, threshold=0.01, max_epochs=1000):
        rmsd = self.get_rmsd()
        epoch = 0
        for epoch in range(max_epochs):
            if np.all(np.less(rmsd, threshold)):
                break

            self.step()
            rmsd = self.get_rmsd()

        return epoch + 1


# Optimization function 1
def q1(pos, max_pos):
    mdist = sqrt(pow(max_pos[0], 2) + pow(max_pos[1], 2)) / 2
    pdist = sqrt(pow((pos[0] - 20), 2) + pow((pos[1] - 7), 2))

    return 100 * (1 - pdist/mdist)


# Optimization function 2
def q2(pos, max_pos):
    mdist = sqrt(pow(max_pos[0], 2) + pow(max_pos[1], 2)) / 2
    pdist = sqrt(pow((pos[0] - 20), 2) + pow((pos[1] - 7), 2))
    ndist = sqrt(pow((pos[0] + 20), 2) + pow((pos[1] + 7), 2))

    return 9 * max(0.0, 10 - pow(pdist, 2)) + 10 * (1 - pdist/mdist) + 70 * (1 - ndist/mdist)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    f = q1
    p = 50
    frames = 150
    
    # create base contour map
    print('Creating contour map')
    x = np.linspace(-50, 50, 100, endpoint=True)
    y = np.linspace(-50, 50, 100, endpoint=True)
    z = np.empty((100, 100))
    for i, x_pos in enumerate(x):
        for j, y_pos in enumerate(y):
            z[j, i] = f((x_pos, y_pos), (-50, 50))

    fig = plt.figure()
    ax = plt.axes(xlim=(-50, 50), ylim=(-50, 50))
    ax.contour(x, y, z, zorder=0)

    scatter = ax.scatter([], [], zorder=1)

    pso = PSO(N=p, obj_func=f, cog=2.0, soc=2.0)

    def init():
        scatter.set_offsets(pso.position.transpose())
        return scatter,

    def animate(i):
        # if i % 10 == 0:
        #     print('Generating frame #{f:d}/{frames:d}'.format(f=i, frames=frames))
        
        pso.step()
        scatter.set_offsets(pso.position.transpose())
        return scatter,
    
    # Create PSO Animation
    print('Animating PSO')
    anim = FuncAnimation(fig, animate, init_func=init, blit=True, frames=frames, interval=20)

    print('Saving animaiton to .gif')
    writergif = PillowWriter(fps=10)
    anim.save('pso_{f_name:s}_action.gif'.format(f_name=f.__name__), writer=writergif)

    print('Displaying Animation')
    plt.show()
