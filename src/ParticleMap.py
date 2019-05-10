import math
import numpy as np


class ParticleMap(object):
    def __init__(self, width=900, height=600, objects=None, number_particles=500):
        self.width = width
        self.height = height

        self.objects = objects

        self.number_particles = number_particles
        self.particles = np.random.rand(number_particles, 3)

        self.particles[:, 0] *= width
        self.particles[:, 1] *= height
        self.particles[:, 2] = self.particles[:, 2] * 2 * math.pi - math.pi

        self.probs = np.ones((self.number_particles,))

    def update_movement(self, linear_mov, rotation, std):
        self.particles[:, 2] += rotation + np.random.rand(self.number_particles) * std[2]

        self.particles[:, 0] += (linear_mov + np.random.rand(self.number_particles) * std[0]) * np.cos(self.particles[:, 2])
        self.particles[:, 1] += (linear_mov + np.random.rand(self.number_particles) * std[1]) * np.sin(self.particles[:, 2])

        particles = np.random.rand(self.number_particles, 3)

        particles[:, 0] *= self.width
        particles[:, 1] *= self.height
        particles[:, 2] = particles[:, 2] * 2 * math.pi - math.pi

        self.particles = np.where(np.logical_or(self.particles[:, 0] < 0, self.particles[:, 0] >= self.width)[..., None],
                                  particles, self.particles)

        self.particles = np.where(np.logical_or(self.particles[:, 1] < 0, self.particles[:, 1] >= self.height)[..., None],
                                  particles, self.particles)

        self.particles[:, 2] = np.where(self.particles[:, 2] > math.pi, - 2 * math.pi + self.particles[:, 2],
                                        self.particles[:, 2])

        self.particles[:, 2] = np.where(self.particles[:, 2] < - math.pi, 2 * math.pi + self.particles[:, 2],
                                        self.particles[:, 2])

    def update_observation(self, observation):
        self.probs = np.ones((self.number_particles,))

        for obs in observation:
            for obj in self.objects:
                if obs[3] == obj[3]:
                    particle_dist = np.linalg.norm(self.particles[:, 0:2] - obj[0:2][None, ...], axis=1)
                    diff = particle_dist - obs[0]
                    variance = np.std(diff) / math.pow(self.number_particles, 1/3)

                    self.probs = self.probs * self.gaussian_error(particle_dist, obs[0], variance)
                    print(self.probs)
                else:
                    pass

        # Normalize probs
        self.probs /= np.sum(self.probs)

        return

    @staticmethod
    def gaussian_error(mu, x, sigma):
        return 1 / (np.sqrt(2*math.pi) * sigma) * np.exp(-0.5 * ((mu - x)/sigma) ** 2)

    def resample(self):
        cumulative_probs_sum = np.cumsum(self.probs)

        # Avoid round-off error
        cumulative_probs_sum[-1] = 1
        indexes = np.searchsorted(cumulative_probs_sum, np.random.random(self.number_particles))

        # Resample according to indexes
        self.particles = self.particles[indexes]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    number_particles = 1000
    # std = [math.sqrt(10/math.sqrt(number_particles)), math.sqrt(10/math.sqrt(number_particles)), math.sqrt(math.pi/math.sqrt(number_particles))]
    std = [10/100, 10/100, math.pi/50]

    robot = [0, 0]

    particle_map = ParticleMap(10, 10, objects=np.array([[10, 0, 0, 1], [5, 5, 0, 2]], dtype=np.float64), number_particles=number_particles)

    plt.scatter(particle_map.particles[:, 0], particle_map.particles[:, 1])
    plt.show()
    plt.clf()

    particle_map.update_movement(1, 0, std)
    particle_map.update_observation(np.array([[9.0, 1, 0, 1], [6.40312, 0, 0, 2]], dtype=np.float64))
    particle_map.resample()

    plt.scatter(particle_map.particles[:, 0], particle_map.particles[:, 1])
    plt.show()
    plt.clf()

    particle_map.update_movement(math.sqrt(2), math.pi / 4, std)

    plt.scatter(particle_map.particles[:, 0], particle_map.particles[:, 1])
    plt.show()
    plt.clf()

    particle_map.update_observation(np.array([[8.06225, 1, 0, 1], [5.0, 0, 0, 2]], dtype=np.float64))
    particle_map.resample()

    plt.scatter(particle_map.particles[:, 0], particle_map.particles[:, 1])
    plt.show()
    plt.clf()

    particle_map.update_movement(math.sqrt(8), 0, std)

    plt.scatter(particle_map.particles[:, 0], particle_map.particles[:, 1])
    plt.show()
    plt.clf()

    particle_map.update_observation(np.array([[6.70820, 1, 0, 1], [2.23606, 0, 0, 2]], dtype=np.float64))
    particle_map.resample()

    plt.scatter(particle_map.particles[:, 0], particle_map.particles[:, 1])
    plt.show()
    plt.clf()

    particle_map.update_movement(math.sqrt(2), 0, std)

    plt.scatter(particle_map.particles[:, 0], particle_map.particles[:, 1])
    plt.show()
    plt.clf()

    particle_map.update_observation(np.array([[6.40312, 1, 0, 1], [1.0, 0, 0, 2]], dtype=np.float64))
    particle_map.resample()

    plt.scatter(particle_map.particles[:, 0], particle_map.particles[:, 1])
    plt.show()
    plt.clf()

    print("ha")
