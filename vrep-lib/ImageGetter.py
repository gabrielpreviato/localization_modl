import vrep.vrep as vrep
import numpy as np


class ImageGetter(object):
    """
    ImageGetter class is responsible for getting an image from a V-REP vision sensor through V-REP Remote API.
    This image will be processed to a numpy array.
    """

    def __init__(self, vrep_connection, vision_sensor_name, mode='blocking'):
        self.vrep_connection = vrep_connection

        self.vision_sensor_name = vision_sensor_name
        self.vision_sensor_handle = vrep_connection.get_object_handle(vision_sensor_name)

        possible_modes = ['blocking']
        self.mode = mode
        if self.mode == 'buffering':
            self.activate_vrep_image_streaming()
        elif self.mode not in possible_modes:
            raise Exception(
                "Impossible to use mode '" + self.mode + "' Possible modes are " + str(possible_modes) + ".")

    def get_image(self):
        if self.mode == 'buffering':
            img, res = self.get_image_vrep_buffering()
        elif self.mode == 'blocking':
            img, res = self.get_image_vrep_blocking()
        else:
            raise Exception("Method for mode '" + self.mode + "' not implemented.")

        return self.vrep_image_to_np(img, res)

    @staticmethod
    def vrep_image_to_np(image, resolution):
        np_src = np.array(image, dtype=np.uint8)
        np_src.resize([resolution[1], resolution[0], 3])
        np_src = np.flipud(np_src)
        return np_src

    def get_image_vrep_blocking(self):
        ret, res, image = vrep.simxGetVisionSensorImage(self.vrep_connection.client_id, self.vision_sensor_handle, 0,
                                                        vrep.simx_opmode_blocking)

        if ret == vrep.simx_return_ok:
            return image, res
        else:
            raise Exception("Failed to get vision sensor image. Vision sensor name: " + self.vision_sensor_name)

    def get_image_vrep_buffering(self):
        ret, res, image = vrep.simxGetVisionSensorImage(self.vrep_connection.client_id, self.vision_sensor_handle, 0,
                                                        vrep.simx_opmode_buffer)

        if ret == vrep.simx_return_ok:
            return image, res
        else:
            raise Exception("Failed to get vision sensor image. Vision sensor name: " + self.vision_sensor_name)

    def activate_vrep_image_streaming(self):
        ret, res, image = vrep.simxGetVisionSensorImage(self.vrep_connection.client_id, self.vision_sensor_handle, 0,
                                                        vrep.simx_opmode_streaming)
        if ret == vrep.simx_return_ok:
            print('Started streaming image')
            return image, res
        elif ret == vrep.simx_return_remote_error_flag or ret == vrep.simx_return_local_error_flag:
            raise Exception("Failed to stream vision sensor image. Vision sensor name: " + self.vision_sensor_name)
        else:
            print('Possible started streaming image')
            return [], [0, 0]
