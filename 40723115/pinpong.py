import pygame

import os , sys
from pygame.examples.moveit import GameObject
from pygame.locals import *

def isCollision( x, y, wallRect):
    if (x >= wallRect[0] and x <= wallRect[0] + wallRect[2] and y >= wallRect[1] and y <= wallRect[1] + wallRect[3]):
        return True;
    return False;
pygame.init()

screen = pygame.display.set_mode((480, 340))
pygame.display.set_caption("air  hockey")
draw_options = pymunk.pygame_util.DrawOptions(screen)
back = pygame.Surface((640, 480))#畫布
background = back.convert()#筆記
background.fill((255, 255, 255))#黑色
goal = pygame.Surface((5, 100))
goal1 = goal.convert()
goal2 = goal.convert()
goal1.fill((0, 255, 0))#球門綠色
goal2.fill((0, 255, 0))
circle = pygame.Surface((15, 15))
circle = background.convert()
circ = pygame.draw.circle(background, (0, 255, 0),(260,170), 15/2 )

#使顏色完全透明，或更準確，使顏色不是 blit。如果你的影象裡面有黑色矩形，你可以設定一個顏色鍵，以防止黑色的顏色。
circle.set_colorkey((255, 255, 255))
circle_x, circle_y = 50, 50
speed_x, speed_y,speed_circ = 500.,500. ,500.
player = pygame.image.load('mallet.jpg')
player_x =120
player_y = 240
x2 = 425
y2 = 50
width = 60
height = 60
vel = 5


clock = pygame.time.Clock()
crashed = False
#white = (255,255,255)

goal = pygame.Surface((5, 100))
goal1 = goal.convert()
goal2 = goal.convert()
goal1.fill((0, 255, 0))#球門綠色
goal2.fill((0, 255, 0))
goal1_x, goal2_x = 5., 470.#球門位置
goal1_y, goal2_y = 130., 130.

#court
RED = pygame.Color(255,0,0)
radius = 50
pygame.draw.line(background,(0,0,0),(240,5),(240,332),1)
pygame.draw.circle(background,(0,0,0),(240,170),radius,1)
wall = pygame.draw.rect(background, (0, 0, 0), Rect((5, 5), (470, 330)), 2)  # 繪製邊
def mallet(x,y):
    screen.blit(player,(x,y))
x = (480 * 0.45)
y = (320* 0.8)
x_change = 0
mallet_speed = 5
def redrawGameWindow():
    screen.blit(mallet, (0, 0))
    pygame.display.update()
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

    keys = pygame.key.get_pressed()    ############################
    if keys[pygame.K_LEFT] and x2 > vel:
        x2 -= vel
    if keys[pygame.K_RIGHT] and x2 < 470-width-vel:
        x2 += vel
    if keys[pygame.K_UP] and y2 > vel:
        y2 -= vel
    if keys[pygame.K_DOWN] and y2 < 330-height-vel:
        y2 += vel
    screen.blit(background, (0, 0))  # 重繪視窗
    screen.blit(goal1, (goal1_x, goal1_y))
    screen.blit(goal2, (goal2_x, goal2_y))
    screen.blit(circle, (circle_x, circle_y))
    mallet(x2, y2)

    time_passed = clock.tick(30)
    time_sec = time_passed / 1000.0

    circle_x += speed_x * time_sec
    circle_y += speed_y * time_sec
    ai_speed = speed_circ * time_sec
    '''
    for wall in range:

        if(isCollision(circ.pos[0], circ.pos[1], wall.rect)):
        '''
            


    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()
