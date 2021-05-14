import pygame,sys,random

def ball_animation():
    global ball_speed_x, ball_speed_y, player_score, opponent_score, score_time
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    if ball.top <= 0 or ball.bottom >= screen_height:
        ball_speed_y *= -1
    if ball.left <= 0:
        player_score +=1
        score_time = pygame.time.get_ticks()

    if ball.right >= screen_width:
        opponent_score += 1
        score_time = pygame.time.get_ticks()
    if ball.colliderect(player) or ball.colliderect(opponent):
        ball_speed_x *= -1
def ball_colliderect(ball, target_rect, colliderect_tolerance):
    global ball_speed_x, ball_speed_y
    if ball.colliderect(target_rect):
        if abs(target_rect.top - ball.bottom) < colliderect_tolerance and ball_speed_y > 0:
            ball_speed_x *= -1
            # print("bottom")
        if abs(target_rect.bottom - ball.top) < colliderect_tolerance and ball_speed_y < 0:
            ball_speed_x *= -1
            # print("top")
        if abs(target_rect.right - ball.left) < colliderect_tolerance and ball_speed_x < 0:
            ball_speed_y *= -1
            # print("left")
        if abs(target_rect.left - ball.right) < colliderect_tolerance and ball_speed_x > 0:
            ball_speed_y *= -1
            # print("right")
def player_animation():#
    player.y += player_speed
    if player.top <= 0:
        player.top = 0
    if player.bottom >= screen_height:
        player.bottom = screen_height
def opponent_ai():
    if opponent.top < ball.y:
        opponent.top += opponent_speed
    if opponent.bottom > ball.y:
        opponent.top -= opponent_speed
    if opponent.top <= 0:
        opponent.top = 0
    if opponent.bottom >= screen_height:
        player.bottom = screen_height
def ball_restart():
    global ball_speed_x, ball_speed_y, score_time

    current_time = pygame.time.get_ticks()
    ball.center = (screen_width/2, screen_height/2)
    if current_time - score_time < 700:
        number_three = game_font.render("3", False, light_grey)
        screen.blit(number_three, (screen_width/2 - 10, screen_height/2 + 20))
    if 700 < current_time - score_time < 1400:
        number_two = game_font.render("2", False, light_grey)
        screen.blit(number_two, (screen_width / 2 - 10, screen_height / 2 + 20))
    if 1400 < current_time - score_time < 2100:
        number_one = game_font.render("1", False, light_grey)
        screen.blit(number_one, (screen_width / 2 - 10, screen_height / 2 + 20))
    if current_time - score_time < 2100:
        ball_speed_x,ball_speed_y = 0, 0
    else:
        ball_speed_y = 7 * random.choice((1, -1))
        ball_speed_x = 7 * random.choice((1, -1))
        score_time = None



pygame.init()
clock = pygame.time.Clock()
#Main window
screen_width = 800
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pong")

ball = pygame.Rect(screen_width/2 - 15, screen_height/2 - 15, 30, 30)
player = pygame.Rect(screen_width - 20, screen_height/2 - 70, 10, 140)
opponent = pygame.Rect(10, screen_height/2 - 70, 10, 140)

bg_color = pygame.Color("grey12")
light_grey = (200, 200, 200)
#Speed setting
ball_speed_x = 7 * random.choice((1, -1))
ball_speed_y = 7 * random.choice((1, -1))
player_speed = 0
opponent_speed = 7
#Score setting
player_score = 0
opponent_score = 0
game_font = pygame.font.Font("freesansbold.ttf", 32)
score_time = True
#Colliderect_tolerance
colliderect_tolerance = 15

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                player_speed += 6
            if event.key == pygame.K_UP:
                player_speed -= 6
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                player_speed -= 6
            if event.key == pygame.K_UP:
                player_speed += 6
    ball_animation()
    player_animation()
    opponent_ai()
    ball_colliderect(ball, player, colliderect_tolerance)
    ball_colliderect(ball, opponent, colliderect_tolerance)

    screen.fill(bg_color)
    pygame.draw.rect(screen, light_grey, player)
    pygame.draw.rect(screen, light_grey, opponent)
    pygame.draw.ellipse(screen, light_grey, ball)
    pygame.draw.aaline(screen, light_grey ,(screen_width/2, 0), (screen_width/2, screen_height))

    if score_time:
        ball_restart()

    player_text = game_font.render(f"{player_score}", False,light_grey)
    screen.blit(player_text, (screen_width/2 + 30, 235))

    opponent_text = game_font.render(f"{opponent_score}", False, light_grey)
    screen.blit(opponent_text, (screen_width/2 - 50, 235))
    #updating the windows
    pygame.display.flip()
    clock.tick(60)