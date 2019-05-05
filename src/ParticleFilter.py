import numpy as np


class ParticleFilter(object):
    def __init__(self, number_classes, particle_map, number_particles=500):
        self.number_classes = number_classes

        self.particle_map = particle_map

        self.number_particles = number_particles
