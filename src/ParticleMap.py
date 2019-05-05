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

    def update_movement(self, linear_mov, rotation):
        self.particles[:, 0:2] += linear_mov
        self.particles[:, 2] += rotation

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


if __name__ == '__main__':
    particle_map = ParticleMap(15, 10, number_particles=10)

    particle_map.update_movement(np.array([1, 1]), np.array([2]))
