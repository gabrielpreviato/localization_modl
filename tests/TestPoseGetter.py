import unittest
import math
import numpy as np

from PoseGetter import PoseGetter
from VrepConnection import VrepConnection


class TestPoseGetter(unittest.TestCase):
    def setUp(self):
        self.vrep_connection = VrepConnection("127.0.0.1", 19997)
        self.vrep_connection.load_scene("scenes/", "test_pose_getter.ttt", absolute_path=False)

        self.test_objects = ['Cuboid', 'Cuboid0', 'Cuboid1']
        self.pose_getters = [PoseGetter(self.vrep_connection, name) for name in self.test_objects]
        self.test_objects_true = np.array([[0.0, 0.0, 0.05, 0.0, 0.0, 0.0], [1.0, 2.0, 0.05, 0.0, 0.0, -math.pi/3],
                                           [-2.0, -2.25, 0.5, 0.0, 0.0, math.pi/4]])

    def test_pose_getter(self):
        for index, pose_getter in enumerate(self.pose_getters):
            np.testing.assert_almost_equal(pose_getter.get_pose(), self.test_objects_true[index], decimal=5)

    def tearDown(self):
        self.vrep_connection.finish_server()


if __name__ == '__main__':
    unittest.main()
