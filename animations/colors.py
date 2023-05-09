from vpython import vec


class Colors:
    """
    A class to manage a color palette to be used consistently in all
    animations.
    """
    def __init__(self):
        """
        The constructor method.
        """
        self.background = vec(235, 235, 235) / 255
        self.red = vec(255, 89, 94) / 255
        self.yellow = vec(255, 202, 58) / 255
        self.green = vec(138, 201, 38) / 255
        self.blue = vec(25, 130, 196) / 255
        self.purple = vec(106, 76, 147) / 255
        self.gray = vec(79, 88, 97) / 255
