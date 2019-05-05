import vrep.vrep as vrep
import numpy as np
import math

from VrepConnection import VrepConnection


class PoseGetter(object):
    """
    PoseGetter class is responsible for getting an object pose from a V-REP object through V-REP Remote API.
    This pose will be processed to a numpy array.
    """

    def __init__(self, vrep_connection, object_name, mode='blocking', reference=-1):
        self.vrep_connection = vrep_connection

        self.object_name = object_name
        self.object_handle = vrep_connection.get_object_handle(object_name)

        possible_modes = ['blocking']
        self.mode = mode
        if self.mode == 'buffering':
            self.activate_vrep_position_streaming()
            self.activate_vrep_orientation_streaming()
        elif self.mode not in possible_modes:
            raise Exception(
                "Impossible to use mode '" + self.mode + "' Possible modes are " + str(possible_modes) + ".")

        self.frame_reference = reference

    def get_pose(self):
        if self.mode == 'buffering':
            pos = self.get_position_vrep_buffering()
            ori = self.get_orientation_vrep_buffering()
        elif self.mode == 'blocking':
            pos = self.get_position_vrep_blocking()
            ori = self.get_orientation_vrep_blocking()
        else:
            raise Exception("Method for mode '" + self.mode + "' not implemented.")

        return self.vrep_pose_to_np(pos, ori)

    @staticmethod
    def vrep_pose_to_np(pos, ori):
        np_pose = np.array(pos + ori, dtype=np.float)
        return np_pose

    def get_position_vrep_blocking(self):
        ret, pos = vrep.simxGetObjectPosition(self.vrep_connection.client_id, self.object_handle, self.frame_reference,
                                              vrep.simx_opmode_blocking)

        if ret == vrep.simx_return_ok:
            return pos
        else:
            raise Exception("Failed to get object position. Object name: " + self.object_name)

    def get_position_vrep_buffering(self):
        ret, pos = vrep.simxGetObjectPosition(self.vrep_connection.client_id, self.object_handle, self.frame_reference,
                                              vrep.simx_opmode_buffer)

        if ret == vrep.simx_return_ok:
            return pos
        else:
            raise Exception("Failed to get object position. Object name: " + self.object_name)

    def activate_vrep_position_streaming(self):
        ret, pos = vrep.simxGetObjectPosition(self.vrep_connection.client_id, self.object_handle, self.frame_reference,
                                              vrep.simx_opmode_streaming)
        if ret == vrep.simx_return_ok:
            print('Started streaming position')
            return pos
        elif ret == vrep.simx_return_remote_error_flag or ret == vrep.simx_return_local_error_flag:
            raise Exception("Failed to stream object position. Object name: " + self.object_name)
        else:
            print('Possible started streaming position')
            return [0, 0, 0]

    def get_orientation_vrep_blocking(self):
        ret, ori = vrep.simxGetObjectOrientation(self.vrep_connection.client_id, self.object_handle,
                                                 self.frame_reference, vrep.simx_opmode_blocking)

        if ret == vrep.simx_return_ok:
            return ori
        else:
            raise Exception("Failed to get object orientation. Object name: " + self.object_name)

    def get_orientation_vrep_buffering(self):
        ret, ori = vrep.simxGetObjectOrientation(self.vrep_connection.client_id, self.object_handle,
                                                 self.frame_reference, vrep.simx_opmode_buffer)

        if ret == vrep.simx_return_ok:
            return ori
        else:
            raise Exception("Failed to get object orientation. Object name: " + self.object_name)

    def activate_vrep_orientation_streaming(self):
        ret, ori = vrep.simxGetObjectOrientation(self.vrep_connection.client_id, self.object_handle,
                                                 self.frame_reference, vrep.simx_opmode_streaming)
        if ret == vrep.simx_return_ok:
            print('Started streaming orientation')
            return ori
        elif ret == vrep.simx_return_remote_error_flag or ret == vrep.simx_return_local_error_flag:
            raise Exception("Failed to stream object orientation. Object name: " + self.object_name)
        else:
            print('Possible started streaming orientation')
            return [0, 0, 0]


if __name__ == '__main__':
    vrep_connection = VrepConnection("127.0.0.1", 19997)
    pose_getter = PoseGetter(vrep_connection, "NAO")

    pose = pose_getter.get_pose()
    pose[3:] *= 180 / math.pi

    print(pose)
