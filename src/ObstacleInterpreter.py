import numpy as np
import itertools

from src.ObjectClasses import ObjectClasses


class ObstacleInterpreter(object):
    def __init__(self, number_classes=4, conf_threshold=0.65, image_width=256, image_height=160, correct_depth=False, iou_threshold=0.5):
        self.number_classes = number_classes
        self.conf_threshold = conf_threshold

        self.image_width = image_width
        self.image_height = image_height

        self.correct_depth = correct_depth

        self.iou_threshold = iou_threshold

    def network_to_corrected_obstacles(self, network_pred):
        # Network predict is of shape 40x(10, 9, 8, 7)
        obstacles = self.network_pred_to_obstacles(network_pred[3])

        if self.correct_depth:
            # correction_factor = self.compute_correction_factor(obstacles, network_pred[0])
            correction_factor = 1.7
            for obstacle in obstacles:
                obstacle[6] *= correction_factor

        return obstacles

    @staticmethod
    def index_to_class(best_class, number_classes):
        if number_classes == 4:
            if best_class == 0 or best_class == 1:
                return best_class + 3
            else:
                return best_class - 1
        elif number_classes == 3:
            if best_class == 0:
                return best_class + 3
            else:
                return best_class
        elif number_classes == 2:
            if best_class == 0:
                return best_class + 3
            else:
                return best_class
        else:
            raise Exception("not implemented")

    def network_pred_to_obstacles(self, prediction):
        def vec_sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if len(prediction.shape) == 2:
            prediction = np.expand_dims(prediction, 0)

        confidence_list = []
        for val in prediction[0, :, 0:self.number_classes]:
            class_confidence = vec_sigmoid(val)

            best_class_index = np.argmax(class_confidence)

            best_class = self.index_to_class(best_class_index, self.number_classes)

            confidence_list.append([class_confidence[best_class_index], ObjectClasses(best_class, self.number_classes)])

        conf = np.asanyarray([i[0] for i in confidence_list], dtype=np.float64)
        conf = np.where(conf > self.conf_threshold, 1, 0)

        x_pos = prediction[0, :, self.number_classes] * conf
        y_pos = prediction[0, :, self.number_classes + 1] * conf
        ws = prediction[0, :, self.number_classes + 2] * conf
        hs = prediction[0, :, self.number_classes + 3] * conf
        mean = prediction[0, :, self.number_classes + 4] * conf * 20.0 * 10            # MODL was trained with normalized means scaled down by 10
        variance = prediction[0, :, self.number_classes + 5] * conf * 20.0 * 10     # MODL was trained with normalized variances scaled down by 10

        detected_obstacles = []
        for i in range(prediction.shape[1]):
            if conf[i] > 0:
                detected_obstacles.append([confidence_list[i][0], confidence_list[i][1].object_class, x_pos[i], y_pos[i], ws[i], hs[i], mean[i], variance[i]])

        return np.array(detected_obstacles)

    def compute_correction_factor(self, obstacles, depth):
        mean_correction = 0
        number_corrections = 0

        for obstacle in obstacles:
            _, _, x, y, ws, hs, mean, _ = obstacle
            x = int(x * self.image_width)
            ws = int(ws * self.image_width)
            y = int(y * self.image_height)
            hs = int(hs * self.image_height)
            mean = mean / 19.75
            depth_roi = depth[0, np.max((y, 0)):np.min((y + hs, depth.shape[1] - 1)),
                              np.max((x, 0)):np.min((x + ws, depth.shape[2] - 1)), 0]

            if len(depth_roi) > 0:
                mean_est = np.mean(depth_roi[(depth_roi > 0) & (depth_roi < 25)])
                if mean_est == np.nan:
                    continue
                number_corrections += 1
                mean_correction += mean_est / mean

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
            if ObjectClasses.class_from_enum(obs[1], self.number_classes) == 'goal':
                obs = np.expand_dims(obs, 0)
                goals = np.append(goals, obs, axis=0)

        goals = np.delete(goals, (0), axis=0)

        permuted_objs = list(itertools.combinations(goals, 2))
        for tup in permuted_objs:
            x_0 = [tup[0][2], tup[0][2] + tup[0][4]]
            x_1 = [tup[1][2], tup[1][2] + tup[1][4]]
            y_0 = [tup[0][3], tup[0][3] + tup[0][5]]
            y_1 = [tup[1][3], tup[1][3] + tup[1][5]]

            iou = self.bb_intersection_over_union([x_0[0], y_0[0], x_0[1], y_0[1]], [x_1[0], y_1[0], x_1[1], y_1[1]])

            if iou > self.iou_threshold:
                # Same obstacle, stay with the one with greater confidence
                if tup[1][0] > tup[0][0]:
                    goals = np.delete(goals, (np.where((goals == tup[0]).all(axis=1))[0]), axis=0)
                else:
                    goals = np.delete(goals, (np.where((goals == tup[1]).all(axis=1))[0]), axis=0)

                permuted_objs = list(itertools.combinations(goals, 2))
        #
        # # We go the best goal posts possible
        # # Calculate x0, x1, y0, y1 from np array
        # goals_easy = np.zeros((goals.shape[0], 6), dtype=np.float64)
        #
        # goals_easy[:, 0] = goals[:, 2] - goals[:, 4] / 2
        # goals_easy[:, 1] = goals[:, 2] + goals[:, 4] / 2
        # goals_easy[:, 2] = goals[:, 3] - goals[:, 5] / 2
        # goals_easy[:, 3] = goals[:, 3] + goals[:, 5] / 2
        # goals_easy[:, 4] = goals[:, 4]
        # goals_easy[:, 5] = goals[:, 5]

        # Vertical posts
        vertical = goals[goals[:, 5] >= goals[:, 4]]

        # Horizontal posts
        horizontal = goals[goals[:, 4] > goals[:, 5]]

        goals_return = []

        # Get the most left vertical post
        try:
            leftest_x = np.argmin(vertical[:, 2], axis=0)
            leftest_x = np.append(vertical[leftest_x], -1)
            goals_return.append(leftest_x)
        except:
            leftest_x = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        # Get the most right vertical post
        try:
            rightest_x = np.argmax(vertical[:, 2], axis=0)
            rightest_x = np.append(vertical[rightest_x], 1)
            goals_return.append(rightest_x)
        except:
            rightest_x = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        # Get the highest (in images y axis is flipped)
        try:
            highest_y = np.argmin(horizontal[:, 3], axis=0)
            highest_y = np.append(horizontal[highest_y], 0)
            goals_return.append(highest_y)
        except:
            highest_y = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        return np.array(goals_return)

    @staticmethod
    def convert_depth_labels_in_meters(png_label):
        # Interpolation obtained via calibration on a test scenario
        metric_depth = -4.586e-09 * (png_label ** 4) + 3.382e-06 * (png_label ** 3) - 0.000105 * (
                png_label ** 2) + 0.04239 * png_label + 0.04072

        return metric_depth

    def obstacles_to_filter(self, interpreted_objs):
        goals = self.goals_to_filter(interpreted_objs)

        if goals.size != 0:
            #goals[:, 6] = self.convert_depth_labels_in_meters(goals[:, 6])
        #     goals[:, 7] /= (19.75 * 100)
            pass

        return goals


if __name__ == "__main__":
    intp = ObstacleInterpreter()

    obstacles = np.array([[0.87, 3, 72, 11, 121, 14, 3.30, 0.0912],
                         [0.76, 3, 73, 14, 8, 85, 2.99, 0.3253],
                         [0.69, 3, 73, 11, 134, 16, 3.21, 0.0897],
                         [0.93, 3, 181, 20, 8, 76, 3.37, 0.0166]])

    print(intp.goals_to_filter(obstacles))
    print(np.max(intp.goals_to_filter(obstacles)))
