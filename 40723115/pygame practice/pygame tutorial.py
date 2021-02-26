import pygame, sys
clock = pygame.time.Clock()#difine clock
from pygame.locals import *
pygame.init()
window_size = (400,400)

screen = pygame.display.set_mode(window_size,0,32)

while True:
#even loop
    for event in pygame.event.get():
        if event.type ==QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    clock.tick(60)#keep the game or the window running at 60 fps