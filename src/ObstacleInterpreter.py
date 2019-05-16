import numpy as np
import itertools

from src.ObjectClasses import ObjectClasses


class ObstacleInterpreter(object):
    def __init__(self, number_classes=4, conf_threshold=0.65, image_width=256, image_height=160, correct_depth=False, iou_threshold=0.25):
        self.number_classes = number_classes
        self.conf_threshold = conf_threshold

        self.image_width = image_width
        self.image_height = image_height

        self.correct_depth = correct_depth

        self.iou_threshold = iou_threshold

    def network_to_corrected_obstacles(self, network_pred):
        # Network predict is of shape 40x(10, 9, 8, 7)
        obstacles = self.network_pred_to_obstacles(network_pred[1])

        if self.correct_depth:
            correction_factor = self.compute_correction_factor(obstacles, network_pred[0])
            for obstacle in obstacles:
                obstacle[1][4] *= correction_factor

        return obstacles

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
                detected_obstacles.append([confidence_list[i][0], confidence_list[i][1].object_class, x_pos[i], y_pos[i], ws[i], hs[i], mean[i], variance[i]])

        return np.array(detected_obstacles)

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

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def goals_to_filter(self, interpreted_objs):
        goals = np.empty((1, 8), dtype=np.float64)
        for obs in interpreted_objs:
            if ObjectClasses.class_from_enum(obs[1], self.number_classes):
                obs = np.expand_dims(obs, 0)
                np.append(goals, obs, axis=0)

        permuted_objs = list(itertools.combinations(goals, 2))
        for tup in permuted_objs:
            x_0 = [tup[0][2] * 32 - tup[0][4] * 256 / 2, tup[0][2] * 32 + tup[0][4] * 256 / 2]
            x_1 = [tup[1][2] * 32 - tup[1][4] * 256 / 2, tup[1][2] * 32 + tup[1][4] * 256 / 2]
            y_0 = [tup[0][3] * 32 - tup[0][5] * 160 / 2, tup[0][3] * 32 + tup[0][5] * 160 / 2]
            y_1 = [tup[1][3] * 32 - tup[1][5] * 160 / 2, tup[1][3] * 32 + tup[1][5] * 160 / 2]

            iou = self.bb_intersection_over_union(x_0 + y_0, x_1 + y_1)

            if iou > self.iou_threshold:
                # Same obstacle, stay with the one with greater confidence
                if tup[1][0] > tup[0][0]:
                    goals = np.delete(goals, (np.where(goals == tup[0])), axis=0)
                else:
                    goals = np.delete(goals, (np.where(goals == tup[1])), axis=0)

        # We go the best goal posts possible
        # Calculate x0, x1, y0, y1 from np array
        goals_easy = np.zeros((goals.shape[0], 6), dtype=np.float64)

        goals_easy[:, 0] = goals[:, 2] * 32 - goals[:, 4] * 256 / 2
        goals_easy[:, 1] = goals[:, 2] * 32 + goals[:, 4] * 256 / 2
        goals_easy[:, 2] = goals[:, 3] * 32 - goals[:, 5] * 160 / 2
        goals_easy[:, 3] = goals[:, 3] * 32 + goals[:, 5] * 160 / 2
        goals_easy[:, 4] = goals[:, 4] * 256
        goals_easy[:, 5] = goals[:, 5] * 160

        # Vertical posts
        vertical = goals_easy[goals_easy[:, 5] > goals_easy[:, 4]]

        # Horizontal posts
        horizontal = goals_easy[goals_easy[:, 4] > goals_easy[:, 5]]

        # Get the most left vertical post
        leftest_x = np.argmin(vertical[:, 0], axis=1)

        # Get the most right vertical post
        rightest_x = np.argmax(vertical[:, 0], axis=1)

        # Get the highest
        highest_y = np.argmax(horizontal[:, 2], axis=1)

        return np.concatenate((leftest_x, rightest_x, highest_y), axis=0)
