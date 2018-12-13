import pickle
import numpy as np

# load bezier curves into numpy arrays
# need to be rotated and normalized to fit with standard graphics top-left axes
def load_glyph(index, scale=1.0, filename='./fonts/arial.quadratic'):
    with open(filename, 'rb') as f:
        # index is 0-61, a-zA-Z0-9
        glyphs = pickle.load(f)
        maximum = np.max(glyphs[index])
        points = np.zeros((len(glyphs[index][0]),len(glyphs[index]),2))
        k = -1
        for i in glyphs[index]:
            k += 1
            for j in range(len(i)):
                points[j][k][0] = glyphs[index][k][j][0] / scale / maximum + (scale - 1) /scale / 2
                points[j][k][1] = 1 - (glyphs[index][k][j][1] / scale / maximum + (scale - 1) / scale / 2)
    return points
