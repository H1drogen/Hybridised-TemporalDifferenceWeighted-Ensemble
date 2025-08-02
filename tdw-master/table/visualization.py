import matplotlib.pyplot as plt
import numpy as np

from env import GOAL, BLOCK, EMPTY


class Visualizer:
    def __init__(self, map_size, num_funcions):
        self.fig, self.axs = plt.subplots(1, 2)

        self.axs[0].set_title('weights')
        self.axs[0].set_ylim(0.0, 1.0)
        self.bar = self.axs[0].bar(np.arange(num_funcions),
                                   np.zeros(num_funcions), animated=True)

        self.axs[1].set_title('map')
        self.map_im = self.axs[1].imshow(np.ones((map_size, map_size)),
                                         cmap='gray', vmin=0.0, vmax=255.0,
                                         animated=True)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axs]
        self.fig.show()

    def _draw_weights(self, weights):
        for i, (rect, w) in enumerate(zip(self.bar.patches, weights)):
            rect.set_height(w)
            self.axs[0].draw_artist(rect)
        self.fig.canvas.blit(self.axs[0].bbox)

    def _draw_map(self, map_table, pos):
        draw_table = map_table.copy()
        draw_table[draw_table == EMPTY] = 255.0
        draw_table[draw_table == BLOCK] = 0.0
        draw_table[pos[0]][pos[1]] = 180.0
        self.map_im.set_data(draw_table)
        self.axs[1].draw_artist(self.map_im)
        self.fig.canvas.blit(self.axs[1].bbox)

    def update(self, map_table, pos, weights):
        for b in self.backgrounds:
            self.fig.canvas.restore_region(b)
        self._draw_weights(weights)
        self._draw_map(map_table, pos)
        self.fig.canvas.flush_events()
