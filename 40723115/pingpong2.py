
import pygame
import random , sys
from pygame.locals import *
import numpy as np
from collections import deque

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SCREEN_SIZE = [320, 400]
BAR_SIZE = [50, 5]
BAR2_SIZE = [50,5]
BALL_SIZE = [15, 15]

# 神經網絡的輸出
MOVE_STAY = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]
colliderect_tolerance = 15
pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Simple Game')
class Game(object):
    def __init__(self):
        self.ball_pos_x = SCREEN_SIZE[0] // 2 - BALL_SIZE[0] / 2
        self.ball_pos_y = SCREEN_SIZE[1] // 2 - BALL_SIZE[1] / 2
        self.ball_pos = pygame.Rect(self.ball_pos_x-5, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
        #self.ball_pos_x, self.ball_pos_y = self.ball_pos.x, self.ball_pos.y

        self.ball_dir_x = -1  # -1 = left 1 = right
        self.ball_dir_y = -1  # -1 = up   1 = down

        self.bar_pos_x = SCREEN_SIZE[0] // 2 - BAR_SIZE[0] // 2
        self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1] - BAR_SIZE[1]-5, BAR_SIZE[0], BAR_SIZE[1])
        #pygame.Rect.colliderect(self.bar_pos,self.ball_pos)

        self.bar2_pos_x = SCREEN_SIZE[0] // 2 - BAR_SIZE[0] // 2
        self.bar2_pos = pygame.Rect(self.bar2_pos_x, 10, BAR_SIZE[0], BAR_SIZE[1])
        self.bar2_speed = 6

        if self.ball_pos.top <= 0 or self.ball_pos.bottom >= SCREEN_SIZE[1]:
            self.ball_start()

    def ball_start(self):
        # global  self.ball_dir_x, ball_pos_y

        self.ball_pos.x = SCREEN_SIZE[0]/2
        self.ball_pos.y = SCREEN_SIZE[1]/2

        print("reset", self.ball_pos.x, self.ball_pos.y)



    # action是MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
    # ai控制棒子左右移動；返回遊戲界面像素數和對應的獎勵。(像素->獎勵->強化棒子往獎勵高的方向移動)


    def step(self, screen):
        '''if action == MOVE_LEFT:
            self.bar_pos_x = self.bar_pos_x - 2
        elif action == MOVE_RIGHT:
            self.bar_pos_x = self.bar_pos_x + 2
        else:
            pass'''
        if self.bar_pos.x < 0:
            self.bar_pos.x = 0
        if self.bar_pos.x > SCREEN_SIZE[0] - BAR_SIZE[0]:
            self.bar_pos.x = SCREEN_SIZE[0] - BAR_SIZE[0]

        if  self.bar2_pos.left < self.ball_pos.x:
             self.bar2_pos.x += self.bar2_speed

        if  self.bar2_pos.right > self.ball_pos.x:
             self.bar2_pos.x -= self.bar2_speed

        if  self.bar2_pos.left <= 0:
             self.bar2_pos.left = 0

        if  self.bar2_pos.right >= SCREEN_SIZE[0]:
             self.bar2_pos.right = SCREEN_SIZE[0]

        screen.fill(BLACK)
        self.bar_pos.left = self.bar_pos_x
        pygame.draw.rect(screen, WHITE, self.bar_pos)
        pygame.draw.rect(screen, WHITE, self.bar2_pos)
        #pygame.draw.rect(screen, (255, 0, 0), Rect((5, 5), (310, 390)), 2)
        pygame.draw.line(screen, (255, 0, 0), (0, 195), (320, 195), 2)
        pygame.draw.circle(screen, (255, 0, 0), (160, 195), 50, 2)

        self.ball_pos.left += self.ball_dir_x * 2
        self.ball_pos.bottom += self.ball_dir_y * 2
        #self.ball_pos.left = 3 * random.choice((1,-1))
        #self.ball_pos.bottom = 3 * random.choice((1,-1))


        pygame.draw.rect(screen, WHITE, self.ball_pos)

        self.colliderect_tolerance = 15
        if self.ball_pos.top <= 0 or self.ball_pos.bottom >= SCREEN_SIZE[1] :
            self.ball_dir_y = self.ball_dir_y * random.choice((1,-1))
            #self.ball_start()

        if self.ball_pos.left <= 0  or self.ball_pos.right >= SCREEN_SIZE[0]:
            self.ball_dir_x = self.ball_dir_x * random.choice((1,-1))


        if self.ball_pos.colliderect(self.bar2_pos) or self.ball_pos.colliderect(self.bar_pos):
            self.ball_dir_y *= -1


game = Game()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    print("pos", game.ball_pos.x, game.ball_pos.y)

    game.step(screen)
    #game.ball_start()
    pygame.display.update()
    clock.tick(60)