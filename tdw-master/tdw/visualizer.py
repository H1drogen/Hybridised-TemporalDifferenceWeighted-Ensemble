import matplotlib.pyplot as plt
import numpy as np


class WeightVisualizer:
    def __init__(self, num_models):
        self.num_models = num_models

        self.fig = plt.figure()
        self.ax = self.fig.gca()

        self.bar = self.ax.bar(np.arange(num_models), np.zeros(num_models),
                               animated=True)
        self.ax.set_title('weights')
        self.ax.set_ylim(0.0, 1.0)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.fig.show()

    def update(self, weights):
        self.fig.canvas.restore_region(self.background)
        for w, rect in zip(weights, self.bar.patches):
            rect.set_height(w)
            self.ax.draw_artist(rect)
        self.fig.canvas.blit(self.ax.bbox)
