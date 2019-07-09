import math
import time

import numpy as np
from matplotlib import pyplot as plt

from ParticleFilter import ParticleFilter
from ParticleMap import ParticleMap
from VrepConnection import VrepConnection
from ImageGetter import ImageGetter
from PoseGetter import PoseGetter
from ObstacleInterpreter import ObstacleInterpreter
from KineticControl import KineticControl


class Robot(object):
    def __init__(self, vrep_connection, image_getter, depth_getter, pose_getter, obstacle_interpreter, particle_filter, network, kinetic_control, show_images=False):
        self.vrep_connection = vrep_connection              # type: VrepConnection
        self.image_getter = image_getter                    # type: ImageGetter
        self.depth_getter = depth_getter
        self.pose_getter = pose_getter                      # type: PoseGetter

        self.network = network

        self.obstacle_interpreter = obstacle_interpreter    # type: ObstacleInterpreter

        self.particle_filter = particle_filter              # type: ParticleFilter

        self.kinetic_control = kinetic_control

        self.mov_it = 10
        self.std = [0.5446159314070489 * self.mov_it * 4, 0.3655749237537081 * self.mov_it * 4, 0.003908048430882346 * self.mov_it]

        self.show_images = show_images

    def run_step(self):
        lin_movement, rotation = 0, 0

        for _ in range(self.mov_it):
            ret = self.kinetic_control.do_movement()
            lin_movement += ret[0]
            rotation += ret[1]
            time.sleep(0.1)

        self.particle_filter.particle_map.update_movement(lin_movement, rotation, self.std)

        plt.scatter(particle_map.particles[:, 0], particle_map.particles[:, 1])

        gt_pose = self.pose_getter.get_pose()
        plt.scatter(gt_pose[0]*100 + 450, gt_pose[1]*100 + 300, c='red')

        plt.ylim(0, 600)
        plt.xlim(0, 900)

        img = self.image_getter.get_image()
        dep = self.depth_getter.get_image()

        # Preprocess img
        img_prep = img.astype(dtype=np.float64) / 255.0

        results = self.network.run(img_prep)

        number_classes = 4

        if self.show_images:
            if results[1] is not None:
                EvaluationUtils.show_detections_multiclass(img, results[1], gt=None, save=False, save_dir="save_"+str(number_classes)+"_symp",
                                                file_name="sav_" + str(i) + ".png", sleep_for=10, multiclass=number_classes)
            if results[0] is not None:
                EvaluationUtils.show_depth(img, results[0], dep, save=False, save_dir="save_"+str(number_classes)+"_symp", file_name="sav_"+ str(i) +".png", sleep_for=10)

        interpreted_objs = self.obstacle_interpreter.network_to_corrected_obstacles(results)

        # TODO change after dataset generator correction
        filter_objs = self.obstacle_interpreter.obstacles_to_filter(interpreted_objs)

        self.particle_filter.particle_map.update_observation(filter_objs)
        self.particle_filter.particle_map.resample()

        prob_x, prob_y = self.particle_filter.particle_map.get_prob_pose()
        plt.scatter(prob_x, prob_y, c='green')

        plt.show()
        plt.clf()



if __name__ == "__main__":
    import lib.EvaluationUtils as EvaluationUtils
    from config import get_config
    # edit config.py as required
    config, unparsed = get_config()

    # Edit model_name to choose model between ['jmod2','cadena','detector','depth','eigen']
    model_name = 'odl'
    number_classes = config.number_classes

    model, detector_only = EvaluationUtils.load_model(model_name, config, number_classes)

    showImages = True

    vrep_connection = VrepConnection("127.0.0.1", 25000, force_finish_comm=False)
    image_getter = ImageGetter(vrep_connection, "NAO_vision1")
    depth_getter = ImageGetter(vrep_connection, "NAO_vision3")
    pose_getter = PoseGetter(vrep_connection, "NAO")
    obstacle_interpreter = ObstacleInterpreter(number_classes=number_classes, correct_depth=False)
    particle_map = ParticleMap(objects=np.array([[900, 430, 2, -1], [900, 170, 2, 1], [900, 300, 2, 0], [900, 250, 1, 0]]), number_particles=2000)
    particle_filter = ParticleFilter(particle_map, number_classes=number_classes)
    kinetic_control = KineticControl()

    robot = Robot(vrep_connection, image_getter, depth_getter, pose_getter, obstacle_interpreter, particle_filter, model, kinetic_control, show_images=True)

    for i in range(10):
        robot.kinetic_control.do_movement()
        time.sleep(0.2)

    print("Finished initial moves")
    for i in range(50):
        robot.run_step()


    plt.scatter(particle_map.particles[:, 0], particle_map.particles[:, 1])
    plt.ylim(0, 600)
    plt.xlim(0, 900)
    plt.show()
    plt.clf()
