
import sys
import pygame
from pygame.locals import *
import random
import tensorflow as tf
import cv2
import numpy as np
from collections import deque
tf.compat.v1.disable_eager_execution()

# 神經網絡的輸出
MOVE_STAY = [1, 0, 0]
MOVE_UP = [0, 1, 0]
MOVE_DOWN = [0, 0, 1]

class Game(object):
   def bouncing_rect(self):
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
               self.blue_start()
               player_score += 1
       if blue.right >=480 and blue.left >= 0:
           if blue.top >= 150 and blue.bottom <=230:
               self.blue_start()
               player2_score += 1
        # Opponent Score

   def blue_start(self):
	   global dx, dy

	   blue.center = (screen_width/2, screen_height/2)
	   dy *= random.choice((1,-1))
	   dx *= random.choice((1,-1))

   def player_animation(self):

       player.y += player_speed

       if player.top <= 0:
           player.top = 0
       if player.bottom >= screen_height:
           player.bottom = screen_height
       if player.left <= 260:
           player.left = 260
       if player.right >= screen_width:
           player.right = screen_width


   def player2_ai(self):

       if player2.top < blue.y:
           player2.y += opponent_speed
       if player2.bottom > blue.y:
           player2.y -= opponent_speed

       if player2.top <= 0:
           player2.top = 0
       if player2.bottom >= screen_height:
           player2.bottom = screen_height


'''
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
'''

# action是MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
    # ai控制棒子左右移動；返回遊戲界面像素數和對應的獎勵。(像素->獎勵->強化棒子往獎勵高的方向移動)
def step(self, action):

    #if action == MOVE_UP:
        self.player_pos_x = self.player_pos_x - 2
    #elif action == MOVE_DOWN:
        self.player_pos_x = self.player_pos_x + 2
    else:
        pass
    if self.player_pos_y < 0:
        self.player_pos_y = 0
    #if self.player_pos_y > screen_SIZE[0] - BAR_SIZE[0]:
       #self.player_pos_y = SCREEN_SIZE[0] - BAR_SIZE[0]

       #self.screen.fill(BLACK)
       #self.player_pos.left = self.player_pos_x
       #pygame.draw.rect(self.screen, WHITE, self.player_pos)

        self.blue_pos.left += self.blue_dir_x * 2
        self.blue_pos.bottom += self.blue_dir_y * 3
       #pygame.draw.rect(self.screen, WHITE, self.ball_pos)

       #if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1] + 1):
           #self.ball_dir_y = self.ball_dir_y * -1
       #if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
           #self.ball_dir_x = self.ball_dir_x * -1

    reward = 0
    if self.bar_pos.top <= self.ball_pos.bottom and (
            self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
           reward = 1  # 擊中獎勵
    elif self.bar_pos.top <= self.ball_pos.bottom and (
            self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
     reward = -1  # 沒擊中懲罰

        # 獲得遊戲界面像素
    screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
    pygame.display.update()
    # 返回遊戲界面像素和對應的獎勵
    return reward, screen_image





pygame.init()
player_speed = 0
opponent_speed = 7

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

# learning_rate
LEARNING_RATE = 0.99
# 更新梯度
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# 測試觀測次數
EXPLORE = 500000
OBSERVE = 50000
# 存儲過往經驗大小
REPLAY_MEMORY = 500000

BATCH = 100

output = 3  # 輸出層神經元數。代表3種操作-MOVE_STAY:[1, 0, 0]  MOVE_LEFT:[0, 1, 0]  MOVE_RIGHT:[0, 0, 1]
input_image = tf.compat.v1.placeholder("float", [None, 80, 100, 4])  # 遊戲像素
action = tf.compat.v1.placeholder("float", [None, output])  # 操作


# 定義CNN-卷積神經網絡 參考:http://blog.topspeedsnail.com/archives/10451
def convolutional_neural_network(input_image):
    weights = {'w_conv1': tf.Variable(tf.zeros([8, 8, 4, 32])),
               'w_conv2': tf.Variable(tf.zeros([4, 4, 32, 64])),
               'w_conv3': tf.Variable(tf.zeros([3, 3, 64, 64])),
               'w_fc4': tf.Variable(tf.zeros([3456, 784])),
               'w_out': tf.Variable(tf.zeros([784, output]))}

    biases = {'b_conv1': tf.Variable(tf.zeros([32])),
              'b_conv2': tf.Variable(tf.zeros([64])),
              'b_conv3': tf.Variable(tf.zeros([64])),
              'b_fc4': tf.Variable(tf.zeros([784])),
              'b_out': tf.Variable(tf.zeros([output]))}

    conv1 = tf.nn.relu(
        tf.nn.conv2d(input_image, weights['w_conv1'], strides=[1, 4, 4, 1], padding="VALID") + biases['b_conv1'])
    conv2 = tf.nn.relu(
        tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1, 2, 2, 1], padding="VALID") + biases['b_conv2'])
    conv3 = tf.nn.relu(
        tf.nn.conv2d(conv2, weights['w_conv3'], strides=[1, 1, 1, 1], padding="VALID") + biases['b_conv3'])
    conv3_flat = tf.reshape(conv3, [-1, 3456])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, weights['w_fc4']) + biases['b_fc4'])

    output_layer = tf.matmul(fc4, weights['w_out']) + biases['b_out']
    return output_layer


# 深度強化學習入門: https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
# 訓練神經網絡
def train_neural_network(input_image, circle_x=dx, circle_y=dy):
    predict_action = convolutional_neural_network(input_image)

    argmax = tf.compat.v1.placeholder("float", [None, output])
    gt = tf.compat.v1.placeholder("float", [None])

    action = tf.reduce_sum(tf.multiply(predict_action, argmax), axis=1)
    cost = tf.reduce_mean(tf.square(action - gt))
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost)

    game = Game()
    D = deque()

    _, image = game.step(MOVE_STAY)
    # 轉換爲灰度值
    image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)
    # 轉換爲二值
    ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    input_image_data = np.stack((image, image, image, image), axis=2)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.initialize_all_variables())

        saver = tf.compat.v1.train.Saver()

        n = 0
        epsilon = INITIAL_EPSILON

        while True:
            for event in pygame.event.get():
             if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            action_t = predict_action.eval(feed_dict={input_image: [input_image_data]})[0]

            argmax_t = np.zeros([output], dtype=np.int)
            if (random.random() <= INITIAL_EPSILON):
                maxIndex = random.randrange(output)
            else:
                maxIndex = np.argmax(action_t)
            argmax_t[maxIndex] = 1
            if epsilon > FINAL_EPSILON:
                 epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    # for event in pygame.event.get():  macOS需要事件循環，否則白屏
    #	if event.type == QUIT:
    #		pygame.quit()
    #		sys.exit()
            reward, image = game.step(list(argmax_t))

            image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
            image = np.reshape(image, (80, 100, 1))
            input_image_data1 = np.append(image, input_image_data[:, :, 0:3], axis=2)

            D.append((input_image_data, argmax_t, reward, input_image_data1))

            if len(D) > REPLAY_MEMORY:
               D.popleft()

            if n > OBSERVE:
                minibatch = random.sample(D, BATCH)
                input_image_data_batch = [d[0] for d in minibatch]
                argmax_batch = [d[1] for d in minibatch]
                reward_batch = [d[2] for d in minibatch]
                input_image_data1_batch = [d[3] for d in minibatch]

                gt_batch = []

                out_batch = predict_action.eval(feed_dict={input_image: input_image_data1_batch})

                for i in range(0, len(minibatch)):
                    gt_batch.append(reward_batch[i] + LEARNING_RATE * np.max(out_batch[i]))

                optimizer.run(feed_dict={gt: gt_batch, argmax: argmax_batch, input_image: input_image_data_batch})

            input_image_data = input_image_data1
            n = n + 1

            if n % 10000 == 0:
                saver.save(sess, 'game.cpk', global_step=n)  # 保存模型

            print(n, "epsilon:", epsilon, " ", "action:", maxIndex, " ", "reward:", reward)

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

    #bouncing_rect()
    #player_animation()
    #player2_animation()

            #pygame.display.update()
            #clock.tick(60)
            train_neural_network(input_image)
#pygame.quit()
#quit()
