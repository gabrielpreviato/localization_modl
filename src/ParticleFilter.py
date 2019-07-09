import numpy as np

from ParticleMap import ParticleMap


class ParticleFilter(object):
    def __init__(self, particle_map, number_classes=4, number_particles=500):
        self.number_classes = number_classes

        self.particle_map = particle_map            # type: ParticleMap

        self.number_particles = number_particles
