import gym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# ball colour = (236,236,236)
# enenmy colour = (213,130,74)
# own colour = (92,186,92)
# background = (144,72,17) or
# backgroun = (109,118,43)

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


env = gym.make('Pong-v0')
env.reset()
img = env.env.render(mode='rgb_array')
orig_img = img
print("image type", type(img))
print("image shape:", img.shape)

'''
_blue = img[y,x,0]
_green = img[y,x,1]
_red = img[y,x,2]
'''

'''
# if need to save as png
env.env.ale.saveScreenPNG(b'test_image.png')
img = mpimg.imread('test_image.png')
'''

# show image using plt
fig = plt.figure()
fig.suptitle('origin image', fontsize=20)
plt.imshow(img)
plt.show()

# show image using cv2

# only get row 35 to 195
img = img[35:195]
fig = plt.figure()
fig.suptitle('img[35:195]', fontsize=20)
print("image shape:", img.shape)
plt.imshow(img)
plt.show()


# background color is (109, 118, 43)
img = img[::2, ::2, 0]
fig = plt.figure()
fig.suptitle('img[::2, ::2, 0]', fontsize=20)
print("image shape:", img.shape)
#plt.imshow(img, cmap='gray')
plt.imshow(img)
plt.show()

'''
img[img == 144] =0
fig = plt.figure()
fig.suptitle('img[img == 144] = 0', fontsize=20)
print("image shape:", img.shape)
#plt.imshow(img, cmap='gray')
plt.imshow(img)
plt.show()
'''

img[img == 109] = 0
fig = plt.figure()
fig.suptitle('img[img == 109] = 0', fontsize=20)
print("image shape:", img.shape)
#plt.imshow(img, cmap='gray')
plt.imshow(img)
plt.show()

img[img !=0] = 1
fig = plt.figure()
fig.suptitle('img[img !=0] = 1', fontsize=20)
print("image shape:", img.shape)
plt.imshow(img, cmap='gray')
#plt.imshow(img)
plt.show()

img =  img.astype(np.float).ravel().reshape(80,80)
fig = plt.figure()
fig.suptitle('flatten', fontsize=20)
print("image shape:", img.shape)
plt.imshow(img, cmap='gray')
#plt.imshow(img)
plt.show()


img_preprocessed = prepro(orig_img).reshape(80,80)
fig = plt.figure()
fig.suptitle('call prepro()', fontsize=20)
plt.imshow(img_preprocessed, cmap='gray')
#plt.imshow(img_preprocessed)
plt.show()

'''
new_img = prepro(img)
print("image type:", type(new_img))
print("image shape:", new_img.shape)
'''

'''
img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
cv2.imshow('image', img)
'''