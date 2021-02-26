import pygame
import os , sys

from pygame.examples.moveit import GameObject
from pygame.locals import *
from sys import exit
import random
pygame.init()

WIN_WIDTH, WIN_HEIGHT = 320, 480
FRAME_PER_SECONDS = 27  # 每秒最大幀數

win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("first game")

img_base_path = os.getcwd() + '/img/'

# 向右走的圖片陣列
malletr = pygame.image.load('mallet.jpg').convert()

# 向左走的圖片陣列
malletl = pygame.image.load('mallet.jpg').convert()
malletl_movex = 0
# 背景
#bg = pygame.image.load('bgg.jpg')
backgroundfile = pygame.image.load('bgg.jpg').convert()

win.blit(backgroundfile, (0, 0))
objects = []
# 站立時的圖片
char = pygame.image.load('mallet.jpg')

x, y = 50, 350  # 起點
width, height = 64, 64  # 寬，高
speed = 5  # 速度

run = True
isJump, left, right = False, False, False
t = 10

walkCount = 0

clock = pygame.time.Clock()

for x in range(10):                    #create 10 objects</i>
    o = GameObject(malletl, x*40, x)
    objects.append(o)
while 1:
    for event in pygame.event.get():
        if event.type in (QUIT, KEYDOWN):
            sys.exit()
    for o in objects:
        win.blit(backgroundfile, o.pos, o.pos)
    for o in objects:
       o.move()
       win.blit(o.image, o.pos)
def redrawGameWindow():
    global walkCount
    win.blit(backgroundfile, (0, 0))

    if walkCount >= FRAME_PER_SECONDS:
        walkCount = 0

    if left:
        # 切換向左走的圖片
        win.blit(malletl, (x, y))
        walkCount += 1
    elif right:
        # 切換向右走的圖片
        win.blit(malletr, (x, y))
        walkCount += 1
    else:
        win.blit(char, (x, y))

        pygame.display.update()


while True:
    clock.tick(FRAME_PER_SECONDS)

    for event in pygame.event.get():
        if event.type == QUIT:
          exit()


        keys = pygame.key.get_pressed()

        if event.type == KEYDOWN:
            keys = pygame.key.get_pressed()
            if keys[K_LEFT]:
                malletl_movex = -speed
    redrawGameWindow()
'''
    elif keys[pygame.K_RIGHT] and x < win.get_size()[0] - width:
        x += speed
        left = False
        right = True
    else:
        left = False
        right = False
        walkCount = 0

    if not isJump:
        if keys[pygame.K_SPACE]:
            isJump = True
            right = False
            left = False
            walkCount = 0
    else:
        if t >= -10:
            a = 1
            if t < 0:
                a = -1
            y -= 0.5 * a * (t ** 2)

            t -= 1
        else:
            isJump = False
            t = 10'''




pygame.quit()