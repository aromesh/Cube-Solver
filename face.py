
class Cubie:
    def __init__(self, color_name, x, y, color_map, id):
        self.color_name = color_name
        self.x = x
        self.y = y
        self.color_map = color_map
        self.id = id

    # Returns rgb color of face
    def get_face_color(self):

        return self.color_map[self.color_name]

    
class Face:
    def __init__(self):
        self.cubies = []
        self.build()

    def build(self, cubie):
        self.cubies.append(cubie)