import pygame, pymunk, sys
# Initial setting
pygame.init()
clock = pygame.time.Clock()
FPS = 60
# Screen setting
screen_width, screen_height = 400, 240
screen = pygame.display.set_mode((screen_width, screen_height))
window_name = "window_name"
pygame.display.set_caption(window_name)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.flip()
    clock.tick(FPS)