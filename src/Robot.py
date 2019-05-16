from ParticleFilter import ParticleFilter
from VrepConnection import VrepConnection
from ImageGetter import ImageGetter
from PoseGetter import PoseGetter
from ObstacleInterpreter import ObstacleInterpreter


class Robot(object):
    def __init__(self, vrep_connection, image_getter, pose_getter, obstacle_interpreter, particle_filter, network):
        self.vrep_connection = vrep_connection              # type: VrepConnection
        self.image_getter = image_getter                    # type: ImageGetter
        self.pose_getter = pose_getter                      # type: PoseGetter

        self.network = network

        self.obstacle_interpreter = obstacle_interpreter    # type: ObstacleInterpreter

        self.particle_filter = particle_filter              # type: ParticleFilter

    def run_step(self):
        img = self.image_getter.get_image_vrep_blocking()

        results = self.network.run(img)

        interpreted_objs = self.obstacle_interpreter.network_to_corrected_obstacles(results)

        filter_objs = self.obstacle_interpreter.obstacles_to_filter(interpreted_objs)
