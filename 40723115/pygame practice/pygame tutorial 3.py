import pygame, sys
clock = pygame.time.Clock()#difine clock
from pygame.locals import *
pygame.init()
window_size = (1000,1000)

screen = pygame.display.set_mode(window_size,0,32)

display = pygame.Surface((1000,900))


player_image = pygame.image.load('mario.jpg')
grass_image = pygame.image.load('grass.png')
dirt_image = pygame.image.load('dirt.png')

moving_right = False
moving_left = False

player_location = [50,50]#人物座標
player_y_momentum = 0
player_rect = pygame.Rect(player_location[0],player_location[1],player_image.get_width(),player_image.get_height())
test_rect = pygame.Rect(100,100,100,50)
while True:
    # the values are for the color you want to fill it
    screen.fill((146,244,255))#RGB#red,green,blue
    screen.blit(player_image,player_location)
#player_image.get_height() which is basically the location of the bottom of the player image
    #if the bottom of the player image is basically touching the bottom of the screen
    if player_location[1] > window_size[1]-player_image.get_height():
        player_y_momentum = -player_y_momentum#flip the momentum
    else:
        player_y_momentum += 0.2
    player_location[1] += player_y_momentum
    if moving_right == True:
        player_location[0] += 4
    if moving_left == True:
        player_location[0] -= 4

    player_rect.x = player_location[0]
    player_rect.y = player_location[1]



#even loop
    for event in pygame.event.get():
        if event.type ==QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:#按下
            if event.key == K_d:
                moving_right = True
            if event.key == K_a:
                moving_left = True
        if event.type == KEYUP:#when the key comes up#放開
            if event.key == K_d:
                moving_right = False
            if event.key == K_a:
                moving_left = False
    surf = pygame.transform.scale(display,window_size)
    screen.blit(surf, (0,0))


    pygame.display.update()
    clock.tick(60)#keep the game or the window running at 60 fps