import pygame, pymunk, sys
# Initial setting
pygame.init()
clock = pygame.time.Clock()
space = pymunk.Space()
FPS = 60
# Mouse trigger
mouse_trigger = False
# Screen setting
screen_width, screen_height = 800, 480
screen = pygame.display.set_mode((screen_width, screen_height))
window_name = "Pymunk"
pygame.display.set_caption(window_name)
# Color
color_bg = (50, 60, 50)
color_ball = (190, 200, 230)
color_payer = (180, 125, 160)
color_line = (210, 190, 189)
# Window collide area
left = 3
right = screen_width - left
top = 3
bottom = screen_height - top
middle_x = screen_width / 2
middle_y = screen_height / 2
center = (middle_x, middle_y)
# Player control
player_up = False
player_down = False
player_left = False
player_right = False
class Ball():
    def __init__(self):
        self.body = pymunk.Body()
        self.body.position = center
        self.body.velocity = 5, -10
        self.radius = 10
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.friction = 1
        self.shape.density = 1
        self.shape.elasticity = 1
        space.add(self.body, self.shape)
        self.shape.collision_type = 1
    def draw(self):
        x, y = self.body.position
        pygame.draw.circle(screen, color_ball, (int(x), int(y)), self.radius)

class Wall():
    def __init__(self, p1 ,p2, collision_number = None):
        self.body = pymunk.Body(body_type = pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, p1, p2, 10)
        self.shape.elasticity = 1
        space.add(self.body, self.shape)
        if collision_number:
            self.shape.collision_type = collision_number

    def draw(self):
        pygame.draw.line(screen, color_line, self.shape.a, self.shape.b, 5)
class Player():
    def __init__(self):
        self.body = pymunk.Body(body_type = pymunk.Body.KINEMATIC)
        self.body.position = right-25, middle_y

        self.radius = 15
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 1
        space.add(self.body, self.shape)
    def draw(self):
        pygame.draw.circle(screen, color_payer, self.body.position, self.radius)
    def move(self, direction):
        v_x, v_y = self.body.velocity
        x, y = self.body.position
        directions = ("UP", "DOWN", "LEFT", "RIGHT")
        velocity = 20
        if direction == directions[0]:
            self.body.velocity = v_x, -velocity
        if direction == directions[1]:
            self.body.velocity = v_x, velocity
        if direction == directions[2]:
            self.body.velocity = -velocity, v_y
        if direction == directions[3]:
            self.body.velocity = velocity, v_y
        # Active area
        if y <= top + self.radius and direction == directions[0]:
            self.body.velocity = v_x, 0
        if y >= bottom - self.radius and direction == directions[1]:
            self.body.velocity = v_x, 0
        if x <= middle_x + self.radius and direction == directions[2]:
            self.body.velocity = 0,v_y
        if x >= right - self.radius and direction == directions[3]:
            self.body.velocity = 0, v_y
    def stop(self):
        self.body.velocity = 0, 0

def airhockey():
    global mouse_trigger,player_up, player_down, player_left, player_right, quit_game
    quit_game = False
    ball = Ball()
    player = Player()
    wall_left = Wall([left, top], [left, bottom])
    wall_right = Wall([right, top], [right, bottom])
    wall_top = Wall([left, top], [right, top])
    wall_bottom = Wall([left, bottom], [right, bottom])
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # Control player
        key = pygame.key.get_pressed()
        if key[pygame.K_UP]:
            #print("up")
            player.move("UP")
        else:
            player.stop()
        if key[pygame.K_DOWN]:
            #print("down")
            player.move("DOWN")
        if key[pygame.K_LEFT]:
            #print("left")
            player.move("LEFT")
        if key[pygame.K_RIGHT]:
            #print("right")
            player.move("RIGHT")
        # Draw object
        screen.fill(color_bg)
        wall_left.draw()
        wall_right.draw()
        wall_top.draw()
        wall_bottom.draw()
        ball.draw()
        player.draw()
        pygame.draw.aaline(screen, color_line, (screen_width / 2, 0), (screen_width / 2, screen_height))
        # Update
        pygame.display.flip()
        space.step(1/FPS)
if __name__ == "__main__":
    airhockey()

