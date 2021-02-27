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
color_bg = (50, 50, 50)
color_ball = (180, 125, 160)
color_payer = (190, 200, 230)
color_line = (210, 190, 189)
# Window collide area
left = 5
right = screen_width - left
top = 5
bottom = screen_height - top
middle_x = screen_width / 2
middle_y = screen_height / 2
class Ball():
    def __init__(self):
        self.body = pymunk.Body()
        self.body.position = middle_x, middle_y
        self.body.velocity = 200, -200
        self.radius = 10
        self.shape = pymunk.Circle(self.body, self.radius)
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
    def __init__(self, x):
        self.body = pymunk.Body(body_type = pymunk.Body.KINEMATIC)
        self.body.position = x, middle_x
        self.shape = pymunk.Segment(self.body, (0, -30), (0, 30), 10)
        self.shape.elasticity = 1
        space.add(self.body, self.shape)
    def draw(self):
        p1 = self.body.local_to_world(self.shape.a)
        p2 = self.body.local_to_world(self.shape.b)
        pygame.draw.line(screen, color_line, p1, p2, 10)
    def move(self, up =True):
        if up:
            self.body.velocity = 0, -600
        else:
            self.body.velocity = 0, 600
    def stop(self):
        self.body.velocity = 0, 0

def game():
    global mouse_trigger
    ball = Ball()
    player = Player(right-15)
    wall_left = Wall([left, top], [left, bottom])
    wall_right = Wall([right, top], [right, bottom])
    wall_top = Wall([left, top], [right, top])
    wall_bottom = Wall([left, bottom], [right, bottom])
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        key = pygame.key.get_pressed()
        if key[pygame.K_UP]:
            player.move()
        elif key[pygame.K_DOWN]:
            player.move(False)
        else:
            player.stop()
        screen.fill(color_bg)
        wall_left.draw()
        wall_right.draw()
        wall_top.draw()
        wall_bottom.draw()
        ball.draw()
        player.draw()
        pygame.draw.aaline(screen, color_line, (screen_width / 2, 0), (screen_width / 2, screen_height))
        pygame.display.flip()
        clock.tick(FPS)
        space.step(1/FPS)
if __name__ == "__main__":
    game()
    pygame.quit()