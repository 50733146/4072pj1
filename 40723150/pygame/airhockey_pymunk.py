import pygame, pymunk, sys, random
# Initial settin
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
color_payer = (240, 250, 10)
color_line = (210, 190, 189)
color_score = (240, 20, 20)
color_text = (240, 20, 20)
# Window collide area
left = 20
right = screen_width - left
top = 30
bottom = screen_height - left
middle_x = screen_width / 2
middle_y = screen_height / 2
center = (middle_x, middle_y)
h = abs(top - bottom)
w = abs(left - right)
def print_text(text, x, y = 5, color = color_text):
    font = pygame.font.Font("freesansbold.ttf", 25)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))


class Ball():
    def __init__(self):
        self.body = pymunk.Body()
        self.body.position = center
        self.body.velocity = 65, -60
        self.radius = 10
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.density = 1
        self.shape.elasticity = 1
        space.add(self.body, self.shape)
        self.shape.collision_type = 1
    def draw(self):
        x, y = self.body.position
        pygame.draw.circle(screen, color_ball, (int(x), int(y)), self.radius)
    def reset(self, space = 0, arbiter = 0, data = 0):
        self.body.position = center
        self.body.velocity = 60 * random.choice((0.7, 1)), -60* random.choice((0.7, 1))
        return False
    def standarize_velocity(self, space = 0, arbiter = 0, data = 0):
        self.body.velocity = self.body.velocity*(50/self.body.velocity.length)
class Wall():
    def __init__(self, p1 ,p2, collision_number = None):
        self.body = pymunk.Body(body_type = pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, p1, p2, 10)
        self.shape.elasticity = 1
        space.add(self.body, self.shape)
        if collision_number:
            self.shape.collision_type = collision_number
    def draw(self, color = color_line, width = 5):
        pygame.draw.line(screen, color, self.shape.a, self.shape.b, width)
class Player():
    def __init__(self, position_x, active_area = "ALL"):
        active_areas = ("ALL", "LEFT", "RIGHT")
        self.body = pymunk.Body(body_type = pymunk.Body.KINEMATIC)
        self.body.position = position_x, middle_y
        self.init_position = self.body.position
        self.radius = 15
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 1
        space.add(self.body, self.shape)
        self.shape.collision_type = 100
        self.score = 0

        if active_area == active_areas[0]:
            self.move_area = 0
        elif active_area == active_areas[1]:
            self.move_area = 1
        elif active_area == active_areas[2]:
            self.move_area = 2

    def draw(self):
        pygame.draw.circle(screen, color_payer, self.body.position, self.radius)
    def move(self, direction):
        v_x, v_y = self.body.velocity
        x, y = self.body.position
        directions = ("UP", "DOWN", "LEFT", "RIGHT")
        velocity = 50
        if direction == directions[0]:
            self.body.velocity = v_x, -velocity
        if direction == directions[1]:
            self.body.velocity = v_x, velocity
        if direction == directions[2]:
            self.body.velocity = -velocity, v_y
        if direction == directions[3]:
            self.body.velocity = velocity, v_y
        # Active area
        if self.move_area == 1:
            if y <= top + self.radius and direction == directions[0]:
                self.body.velocity = v_x, 0
                self.body.position = x, top + self.radius
            if y >= bottom - self.radius and direction == directions[1]:
                self.body.velocity = v_x, 0
                self.body.position = x, bottom - self.radius
            if x <= left + self.radius and direction == directions[2]:
                self.body.velocity = 0, v_y
                self.body.position = left + self.radius, y
            if x >= middle_x - self.radius and direction == directions[3]:
                self.body.velocity = 0, v_y
                self.body.position = middle_x - self.radius, y
        elif self.move_area == 2:
            if y <= top + self.radius and direction == directions[0]:
                self.body.velocity = v_x, 0
                self.body.position = x, top + self.radius
            if y >= bottom - self.radius and direction == directions[1]:
                self.body.velocity = v_x, 0
                self.body.position = x, bottom - self.radius
            if x <= middle_x + self.radius and direction == directions[2]:
                self.body.velocity = 0, v_y
                self.body.position = middle_x + self.radius, y
            if x >= right - self.radius and direction == directions[3]:
                self.body.velocity = 0, v_y
                self.body.position = right - self.radius, y
    def stop(self):
        self.body.velocity = 0, 0




def airhockey():
    ball = Ball()
    wall_left = Wall([left, top], [left, bottom])
    wall_right = Wall([right, top], [right, bottom])
    wall_top = Wall([left, top], [right, top])
    wall_bottom = Wall([left, bottom], [right, bottom])
    # Player
    player_1 = Player(left+15, "LEFT")
    player_2 = Player(right-15, "RIGHT")

    contact_with_player = space.add_collision_handler(1, 100)
    contact_with_player.post_solve = ball.standarize_velocity
    # Score
    score_1 = Wall([right, top + h/3], [right, bottom - h/3], 101)
    score_2 = Wall([left, top + h/3], [left, bottom - h/3], 102)
    scored_1 = space.add_collision_handler(1, 101)
    scored_2 = space.add_collision_handler(1, 102)

    def player1_scored(space, arbiter, data):
        player_1.score += 1
        #print(player_1.score)
        ball.reset()
        return False
    scored_1.begin = player1_scored

    def player2_scored(space, arbiter, data):
        player_2.score += 1
        ball.reset()
        return False
    scored_2.begin = player2_scored

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # Control player
        key = pygame.key.get_pressed()
        # Player 1
        if key[pygame.K_w]:
            #print("up")
            player_1.move("UP")
        else:
            player_1.stop()
        if key[pygame.K_s]:
            #print("down")
            player_1.move("DOWN")
        if key[pygame.K_a]:
            #print("left")
            player_1.move("LEFT")
        if key[pygame.K_d]:
            #print("right")
            player_1.move("RIGHT")
        # Player 2
        if key[pygame.K_UP]:
            # print("up")
            player_2.move("UP")
        else:
            player_2.stop()
        if key[pygame.K_DOWN]:
            #print("down")
            player_2.move("DOWN")
        if key[pygame.K_LEFT]:
            #print("left")
            player_2.move("LEFT")
        if key[pygame.K_RIGHT]:
            #print("right")
            player_2.move("RIGHT")


        # Draw object
        screen.fill(color_bg)
        wall_left.draw()
        wall_right.draw()
        wall_top.draw()
        wall_bottom.draw()
        ball.draw()
        player_1.draw()
        player_2.draw()
        pygame.draw.line(screen, color_line, (screen_width / 2, top), (screen_width / 2, bottom))
        print_text(f"P1 : {player_1.score}", left)
        print_text(f"P2 : {player_2.score}", right-90)
        # Score
        score_1.draw(color=color_score, width=10)
        score_2.draw(color=color_score, width=10)
        # Update
        pygame.display.flip()
        space.step(1/FPS)
if __name__ == "__main__":
    airhockey()

