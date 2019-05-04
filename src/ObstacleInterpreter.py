import numpy as np

from src.ObjectClasses import ObjectClasses


class ObstaclesInterpreter(object):
    def __init__(self, number_classes=4, conf_threshold=0.65, image_width=256, image_height=160, correct_depth=False):
        self.number_classes = number_classes
        self.conf_threshold = conf_threshold

        self.image_width = image_width
        self.image_height = image_height

        self.correct_depth = correct_depth

    def network_to_filter(self, network_pred):
        # Network predict is of shape 40x(10, 9, 8, 7)
        obstacles = self.network_pred_to_obstacles(network_pred[1])

        if self.correct_depth:
            correction_factor = self.compute_correction_factor(obstacles, network_pred[0])
            for obstacle in obstacles:
                obstacle[1][4] *= correction_factor

    def network_pred_to_obstacles(self, prediction):
        def vec_sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if len(prediction.shape) == 2:
            prediction = np.expand_dims(prediction, 0)

        confidence_list = []
        for val in prediction[0, :, 0:self.number_classes]:
            class_confidence = vec_sigmoid(val)

            best_class = np.argmax(class_confidence)
            confidence_list.append([class_confidence[best_class], ObjectClasses(best_class, self.number_classes)])

        conf = np.asanyarray([i[0] for i in confidence_list], dtype=np.float64)
        conf = np.where(conf > self.conf_threshold, 1, 0)

        x_pos = prediction[0, :, self.number_classes] * conf
        y_pos = prediction[0, :, self.number_classes + 1] * conf
        ws = prediction[0, :, self.number_classes + 2] * conf
        hs = prediction[0, :, self.number_classes + 3] * conf
        mean = prediction[0, :, self.number_classes + 4] * conf * 19.75 * 10            # MODL was trained with normalized means scaled down by 10
        variance = prediction[0, :, self.number_classes + 5] * conf * 19.75 * 1000      # MODL was trained with normalized variances scaled down by 1000

        detected_obstacles = []
        for i in range(prediction.shape[1]):
            if conf[i] > 0:
                detected_obstacles.append([confidence_list[i], [x_pos[i], y_pos[i], ws[i], hs[i], mean[i], variance[i]]])

        return detected_obstacles

    @staticmethod
    def compute_correction_factor(obstacles, depth):
        mean_correction = 0
        number_corrections = 0

        for obstacle in obstacles:
            x, y, ws, hs, mean, _ = obstacles[1]
            depth_roi = depth[0, np.max((y - hs / 2, 0)):np.min((y + hs / 2, depth.shape[1] - 1)),
                              np.max((x - ws / 2, 0)):np.min((x + ws / 2, depth.shape[2])), 0]

            if len(depth_roi) > 0:
                mean_est = np.mean(depth_roi)
                number_corrections += 1
                mean_correction += mean_est / obstacle.mean

            if number_corrections > 0:
                mean_correction /= number_corrections
            else:
                mean_correction = 1

            return mean_correction
