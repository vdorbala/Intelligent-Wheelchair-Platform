# Detecting click events for annotation
import numpy as np
import matplotlib.pyplot as plt
coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print 'x = %d, y = %d'%(
        ix, iy)

    global coords
    coords.append((ix, iy))

    if len(coords) == 1:
        fig.canvas.mpl_disconnect(cid)

    return coords

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
