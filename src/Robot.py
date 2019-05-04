from VrepConnection import VrepConnection
from ImageGetter import ImageGetter
from PoseGetter import PoseGetter


class Robot(object):
    def __init__(self, vrep_connection, image_getter, pose_getter, obstacle_interpreter, particle_filter):
        self.vrep_connection = vrep_connection
        self.image_getter = image_getter
        self.pose_getter = pose_getter

        self.obstacle_interpreter = obstacle_interpreter

        self.particle_filter = particle_filter
