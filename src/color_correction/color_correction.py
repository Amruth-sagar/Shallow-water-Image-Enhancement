from adaptive_stretch_lab import LAB_Stretching

class ColorCorrection():
    @classmethod
    def color_correct(cls, image):
        return LAB_Stretching(image)


