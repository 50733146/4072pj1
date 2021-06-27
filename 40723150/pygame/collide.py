import pygame,sys

def ball_animation():
    global ball_x, ball_y, ball_speed_x, ball_speed_y
    ball_x += ball_speed_x
    ball_y += ball_speed_y
    ball.x, ball.y = ball_x, ball_y
    if ball.top <= 0 or ball.bottom >= screen_height:
        ball_speed_y *= -1
    if ball.left <= 0 or ball.right >= screen_width:
        ball_speed_x *= -1
    if abs(ball_speed_x) >= 8:
        if abs(ball_speed_x) >= 10:
            ball_speed_x = 10
        ball_speed_x -= 0.8
    if abs(ball_speed_y) >= 8:
        if abs(ball_speed_y) >= 10:
            ball_speed_y = 10
        ball_speed_y -= 0.8
    #print(ball_speed_x, ball_speed_y)
def ball_start():
    global ball_x, ball_y, ball_init_x, ball_init_y
    ball_x, ball_y = ball_init_x, ball_init_y
def player_animation():
    global mouse_trigger, player_speed_x, playerspeed_y
    if mouse_trigger == True:
        player_x, player_y = pygame.mouse.get_pos()
        player_speed_x, playerspeed_y = pygame.mouse.get_rel()
        player.x, player.y = player_x-15, player_y-15

def ball_colliderect(colliderect_tolerance):
    global ball_speed_x, ball_speed_y, player_speed_x, player_speed_y
    if ball.colliderect(player):
        #print("true")
        #print("ball :\n"+"top:"+str(ball.top)+" "+"bottom:"+str(ball.bottom)+"\n"+"left:"+str(ball.left)+" "+"right:"+str(ball.right))
        #print("player :\n"+"top:"+str(player.top)+" "+"bottom:"+str(player.bottom)+"\n"+"left:"+str(player.left)+" "+"right:"+str(player.right))
        if player.bottom - ball.top <= colliderect_tolerance and ball.y >= player.y:
            ball_speed_y *= -1
            ball_speed_y += player_speed_y
            print("top collide")
        if ball.bottom - player.top >= colliderect_tolerance and ball.y <= player.y:
            ball_speed_y *= -1
            ball_speed_y += player_speed_y
            print("bottom collide")
        #print(ball.left, ball.right)
        #print(player.left, player.right)
def ball_colliderect_c(colliderect_tolerance):
    global ball_speed_x, ball_speed_y, player_speed_x, player_speed_y
    if ball.colliderect(player):
        #print("true")
        #print("ball :\n"+"top:"+str(ball.top)+" "+"bottom:"+str(ball.bottom)+"\n"+"left:"+str(ball.left)+" "+"right:"+str(ball.right))
        #print("player :\n"+"top:"+str(player.top)+" "+"bottom:"+str(player.bottom)+"\n"+"left:"+str(player.left)+" "+"right:"+str(player.right))
        if ball.top <= player.bottom and ball.y >= player.y:
            ball_speed_y *= -1
            ball_speed_y = abs(ball_speed_y) - 1
            ball_speed_y += player_speed_y
            print("top collide")
        if ball.bottom >= player.top and ball.y <= player.y:
            ball_speed_y *= -1
            ball_speed_y = abs(ball_speed_y) - 1
            ball_speed_y += player_speed_y
            print("bottom collide")
        #print(ball.left, ball.right)
        #print(player.left, player.right)
        if ball.right >= player.left and ball.x <= player.x:
            ball_speed_x *= -1
            ball_speed_x = abs(ball_speed_x) - 1
            ball_speed_x += player_speed_x
            print("right collide")
        if ball.left <= player.right and ball.x >= player.x:
            ball_speed_x *= -1
            ball_speed_x = abs(ball_speed_x) - 1
            ball_speed_x += player_speed_x
            print("left collide")
def ball_colliderect_a(colliderect_tolerance):
    global ball_speed_x, ball_speed_y, player_speed_x, player_speed_y
    if ball.colliderect(player):
        print("true")
        if ball.bottom - player.top < colliderect_tolerance and abs(ball_speed_y) > 0:
            ball_speed_x *= -1
            ball_speed_x = abs(ball_speed_x) - 1
            ball_speed_x += player_speed_x
            print("bottom")
        if player.bottom - ball.top< colliderect_tolerance and abs(ball_speed_y) > 0:
            ball_speed_x *= -1
            ball_speed_x = abs(ball_speed_x) - 1
            ball_speed_x += player_speed_x
            print("top")
        if player.right - ball.left < colliderect_tolerance and abs(ball_speed_x) > 0:
            ball_speed_y *= -1
            ball_speed_y = abs(ball_speed_y) - 1
            ball_speed_y += player_speed_y
            print("left")
        if ball.right - player.left < colliderect_tolerance and abs(ball_speed_x) > 0:
            ball_speed_y *= -1
            ball_speed_y = abs(ball_speed_y) - 1
            ball_speed_y += player_speed_y
            print("right")


pygame.init()
clock = pygame.time.Clock()
# Mouse trigger
mouse_trigger = False
# Main window
screen_width = 400
screen_height = 240
screen = pygame.display.set_mode((screen_width, screen_height))
# Window name
pygame.display.set_caption("Collide")
# Color setting
color_bg = (50, 60, 50)
color_ball = (120, 150 ,160)
color_player = (170, 125, 120)
# Game object
ball_init_x, ball_init_y = screen_width/2 - 15, screen_height/2 - 15
ball = pygame.Rect(ball_init_x, ball_init_y, 30, 30)
player = pygame.Rect(screen_width*3/4 - 15, screen_height/2 - 15, 30, 30)
# Speed setting
ball_speed_x = 7
ball_speed_y = 7
player_speed_x = 0
player_speed_y  = 0
# Position
ball_x, ball_y = ball_init_x, ball_init_y
ball_position = [ball_x, ball_y]
#ball_start()
while True:
    for event in pygame.event.get():
        # get events from the queue
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_trigger = not mouse_trigger
            #print(mouse_trigger)
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_trigger = not mouse_trigger
            #print(mouse_trigger)
    ball_animation()
    player_animation()
    ball_colliderect(15)
    screen.fill(color_bg)
    pygame.draw.ellipse(screen, color_ball, ball)
    pygame.draw.ellipse(screen, color_player, player)
    #Update the full display Surface to the screen
    pygame.display.flip()
    clock.tick(60)