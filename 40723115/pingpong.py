
import sys
import pygame
from pygame.locals import *
import random

def bouncing_rect():
    global dx , dy ,player_x,player_y,player_score,player2_score,goal1_x,goal1_y

    blue.x += dx
    blue.y += dy

    if blue.top <= 0 or blue.bottom >=340:
        dy *= -1
    if blue.left <= 0 or blue.right >= 480:
        dx *= -1
    if blue.colliderect(player) or blue.colliderect(player2):
        dx *= -1
        # Player Score
        # Player Score
    if blue.left <= 0 and blue.right <= 480:
        if blue.top >= 150 and blue.bottom >=230:
           blue_start()
           player_score += 1
    if blue.right >=480 and blue.left >= 0:
        if blue.top >= 150 and blue.bottom <=230:
            blue_start()
            player2_score += 1
        # Opponent Score

def blue_start():
	global dx, dy

	blue.center = (screen_width/2, screen_height/2)
	dy *= random.choice((1,-1))
	dx *= random.choice((1,-1))

def player_animation():

    player.y += player_speed
    # player.x += player_speed
    if player.top <= 0:
        player.top = 0
    if player.bottom >= screen_height:
        player.bottom = screen_height
    if player.left <= 260:
        player.left = 260
    if player.right >= screen_width:
        player.right = screen_width

def player2_animation():

    player2.y += player2_speed
   # player.x += player_speed
    if player2.top <= 0:
        player2.top = 0
    if player2.bottom >= screen_height:
        player2.bottom = screen_height
    if player2.left <= 260:
        player2.left = 260
    if player2.right >= 20:
        player2.right = 20

pygame.init()
player_speed = 0
player2_speed = 0

circle_x , circle_y = 300,200

screen_width = 480
screen_height = 340

player_score = 0
player2_score = 0
font = pygame.font.SysFont("calibri", 40)

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("air hockey")

clock = pygame.time.Clock()

background = pygame.Surface((640, 480))#畫布
background = background.convert()#筆記
background.fill((255, 255, 255))

goal = pygame.Surface((5, 100))
goal1 = goal.convert()
goal2 = goal.convert()
goal1.fill((0, 255, 0))#球門綠色
goal2.fill((0, 255, 0))
goal1_y, goal2_y = 130., 130.
goal1_x, goal2_x = 5., 470.#球門位置


player = pygame.Rect(480 - 20, 340 / 2 - 70, 10, 70)
blue = pygame.Rect(480 / 2 - 15, 340 / 2 - 15, 30, 30)
player2 = pygame.Rect(20, screen_height / 2 - 70, 10, 70)



RED = pygame.Color(255,0,0)
radius = 50
pygame.draw.line(background,(0,0,0),(240,5),(240,332),10)
pygame.draw.circle(background,(0,0,0),(240,170),radius,10)

# 創建藍色小球(它也有自己的畫布，移動球即等於移動它的畫布)
ball = pygame.Surface((30, 30))  # 建立球矩形繪圖區
ball.fill((255, 255, 255))  # 這邊我故意將球的背景色設為黃色，以清楚看到球的畫布
blue = pygame.draw.circle(ball, (0, 0, 255), (15, 15), 15, 0)  # 畫藍色球
rect1 = ball.get_rect()  # 取得球矩形區塊
rect1.center = (320, 170)  # 球起始中心位置
x, y = rect1.topleft  # 球左上角坐標
dx = 4 * random.choice((1, -1))
dy = 4 * random.choice((1, -1))# 球運動速度
white = (255,255,255)

wall = pygame.draw.rect(background, (0, 0, 0), Rect((5, 5), (470, 330)), 10)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                player_speed -= 6
            if event.key == pygame.K_DOWN:
                player_speed += 6
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                player_speed += 6
            if event.key == pygame.K_DOWN:
                player_speed -= 6

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                player2_speed -= 6
            if event.key == pygame.K_s:
                player2_speed += 6
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                player2_speed += 6
            if event.key == pygame.K_s:
                player2_speed -= 6

    score1 = font.render(str(player_score), True, (255, 0, 255))
    score2 = font.render(str(player2_score), True, (255, 0, 255))

    screen.blit(background, (0, 0))  # 重繪視窗
    screen.blit(goal1, (goal1_x, goal1_y))
    screen.blit(goal2, (goal2_x, goal2_y))
    screen.blit(score1, (255., 155))
    screen.blit(score2, (205., 155.))
    background.fill((255, 255, 255))

    pygame.draw.line(background, (0, 0, 0), (240, 5), (240, 332), 10)
    pygame.draw.circle(background, (0, 0, 0), (240, 170), radius, 10)
    pygame.draw.rect(background, (0, 0, 0), Rect((5, 5), (470, 330)), 10)
    pygame.draw.rect(background, (255,0,0), player)
    pygame.draw.rect(background, (255,0,0), player2)
    pygame.draw.ellipse(background, (0,0,255), blue)



    circle_x += dx
    circle_y += dy

    bouncing_rect()
    player_animation()
    player2_animation()

    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()
