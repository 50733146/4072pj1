import gym
import matplotlib.pyplot as plt

env = gym.make('Pong-v0')
env.reset()

img = env.render(mode='rgb_array')
orig_img = img
print("image type", type(img))
print("image shape:", img.shape)

# show image using plt
fig = plt.figure()
fig.suptitle('origin image', fontsize=20)
plt.imshow(img)
plt.show()
env.close()