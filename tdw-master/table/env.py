import numpy as np


EMPTY = 0
BLOCK = 1
GOAL = 2


UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3


class GridWorld:
    def __init__(self, size=3, goal_reward=1.0, step_penalty=0.0, limit=100):
        self.size = size
        self.step_penalty = step_penalty
        self.limit = limit
        self.map = np.zeros((size, size), dtype=np.uint8)
        self.pos = (0, 0)
        self.goal_reward = goal_reward
        self.t = 0
        self._setup_map()
        self._reset_pos()

    def _setup_map(self):
        raise NotImplementedError

    def _reset_pos(self):
        raise NotImplementedError

    def _get_index(self):
        return self.size * self.pos[0] + self.pos[1]

    def _movable(self, pos):
        if pos[0] < 0 or pos[1] < 0:
            return False
        elif pos[0] > self.size - 1 or pos[1] > self.size - 1:
            return False
        elif self.map[pos[0]][pos[1]] == BLOCK:
            return False
        return True

    def move(self, action):
        if action == UP:
            pos = (self.pos[0] - 1, self.pos[1])
        elif action == DOWN:
            pos = (self.pos[0] + 1, self.pos[1])
        elif action == RIGHT:
            pos = (self.pos[0], self.pos[1] + 1)
        elif action == LEFT:
            pos = (self.pos[0], self.pos[1] - 1)
        if self._movable(pos):
            self.pos = pos

    def _reward(self):
        if self.map[self.pos[0]][self.pos[1]] == GOAL:
            return self.goal_reward
        return self.step_penalty

    def _done(self):
        if self.map[self.pos[0]][self.pos[1]] == GOAL:
            return True
        if self.t == self.limit - 1:
            return True
        return False

    def step(self, action):
        self.move(action)
        reward = self._reward()
        obs = self._get_index()
        done = self._done()
        self.t += 1
        return obs, reward, done, {}

    def reset(self):
        self._reset_pos()
        self.t = 0
        return self._get_index()


class CenterBlockEnv(GridWorld):
    def _setup_map(self):
        self.map[0][self.size // 2] = GOAL
        self.map[1:self.size - 1, 1:self.size - 1] = BLOCK

    def _reset_pos(self):
        self.pos = (self.size - 1, self.size // 2)


class SlitEnv(GridWorld):
    def _setup_map(self):
        self.map[0][self.size // 2] = GOAL
        self.map[self.size // 2, 1:self.size - 1] = BLOCK

    def _reset_pos(self):
        self.pos = (self.size - 1, self.size // 2)


class DoubleSlitEnv(GridWorld):
    def _setup_map(self):
        self.map[0][self.size // 2] = GOAL
        self.map[-1][self.size // 2] = GOAL
        self.map[self.size // 4, 1:self.size - 1] = BLOCK
        self.map[self.size // 4 * 3, 1:self.size - 1] = BLOCK

    def _reset_pos(self):
        self.pos = (self.size // 2, self.size // 2)


class Grid20x20(GridWorld):
    def __init__(self,
                 start=(0, 19),
                 goal=(19, 0),
                 goal_reward=1.0,
                 step_penalty=0.0,
                 limit=1000):
        self.start = start
        self.goal = goal
        super().__init__(20, goal_reward=goal_reward,
                         step_penalty=step_penalty, limit=limit)

    def _setup_map(self):
        self.map[self.goal[1]][self.goal[0]] = GOAL

    def _reset_pos(self):
        self.pos = (self.start[1], self.start[0])
