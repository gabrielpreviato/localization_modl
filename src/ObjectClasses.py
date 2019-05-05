class ObjectClasses(object):
    def __init__(self, object_class, number_classes):
        self.object_class = object_class + (number_classes - 2) * 4

        if number_classes == 2 or number_classes == 3:
            if object_class == 0:
                self.object = 'robot'
            elif object_class == 1:
                self.object = 'ball'
            elif object_class == 2:
                self.object = 'goal'
        elif number_classes == 4:
            if object_class == 0:
                self.object = 'robot_team'
            elif object_class == 1:
                self.object = 'robot_opponent'
            elif object_class == 2:
                self.object = 'ball'
            elif object_class == 3:
                self.object = 'goal'
        else:
            raise Exception("There is no class for the combination object_class == " + str(object_class) +
                            " and number_classes == " + str(number_classes) + ".")
