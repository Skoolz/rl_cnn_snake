import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch
import gymnasium as gym
from collections import deque
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI(gym.Env):

    def __init__(self, w=160, h=160):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.b_w = round(w/BLOCK_SIZE)
        self.b_h = round(h/BLOCK_SIZE)
        self.stacked_state = deque(maxlen=4)
        self.reset()
        self.action_space = Discrete(4)
        self.observation_space = Box(0,255,(4,8,8),dtype=np.uint8)


    def get_state(self):
        game_board = self.get_map()

        while(len(self.stacked_state)<4):
            self.stacked_state.append(game_board)
        
        self.stacked_state.append(game_board)

        return torch.cat(list(self.stacked_state)) #(C,W,H)


    def reset(self,seed=None, options=None):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.stacked_state = deque(maxlen=4)
        self.old_dist = self.get_dist(self.head,self.food)

        info = {}
        return self.get_state(),info


    def _place_food(self):
        x = random.randint(1, (self.w-BLOCK_SIZE )//BLOCK_SIZE -1)*BLOCK_SIZE
        y = random.randint(1, (self.h-BLOCK_SIZE )//BLOCK_SIZE -1)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False

        dist = self.get_dist(self.head,self.food)
        reward = 0.1 * (self.old_dist-dist)
        self.old_dist = dist
        info = {'score':self.score}

        if self.is_collision():
            game_over = True
            reward = -10
            return self.get_state(), reward, game_over,False,info
        
        if(self.frame_iteration > 100*len(self.snake)):
            game_over = True
            reward = -10
            return self.get_state(),reward,game_over,True,info

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self.render()
        self.clock.tick(SPEED)
        # 6. return game over and score
        info = {'score':self.score}

        return self.get_state(), reward, game_over, False, info


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def render(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def close(self):
        pygame.quit()
        quit()


    def _move(self, action):
        # [straight, right, left]

        if(action==0):
            new_dir = Direction.LEFT
        elif(action==1):
            new_dir = Direction.RIGHT
        elif(action==2):
            new_dir = Direction.UP
        else:
            new_dir = Direction.DOWN

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
    
    def get_coord(self,p):
        return Point(round(p.x/BLOCK_SIZE),round(p.y/BLOCK_SIZE))
    
    def get_map(self):
        game_map = torch.zeros((self.b_w,self.b_h),dtype=torch.float)

        for i in range(self.b_w):
            for j in range(self.b_h):
                if(i==0 or i==self.b_w-1 or j == 0 or j == self.b_h-1):
                    game_map[i][j]=1
        
        for p in self.snake[1:]:
            p_p = self.get_coord(p)
            game_map[p_p.x][p_p.y] = 1
        
        h_p = self.get_coord(self.head)
        if(h_p.x >= 0 and h_p.x <= self.b_w-1 and h_p.y >= 0 and h_p.y <= self.b_h-1):
            game_map[h_p.x][h_p.y]=2

        f_p = self.get_coord(self.food)
        game_map[f_p.x][f_p.y]=-20

        return game_map.unsqueeze(0)
    
    def get_dist(self,p1,p2):
        p1 = self.get_coord(p1)
        p2 = self.get_coord(p2)

        return (p1.x-p2.x)**2 + (p1.y-p2.y)**2