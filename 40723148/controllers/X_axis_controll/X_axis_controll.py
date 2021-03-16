from controller import Robot, Camera, Motor
import cv2

robot = Robot()
timestep = 64

camera = robot.getCamera('camera')
Camera.enable(camera, timestep)

Xspeed = 0.0

Xaxis = robot.getMotor('Xaxis')
Xaxis.setPosition(float('inf'))

while robot.step(timestep) != -1:
    k = cv2.waitKey(10) & 0xFF
    Camera.getImage(camera)
    Camera.saveImage(camera, 'img.png', 1)
    frame = cv2.imread('img.png')
    frame = cv2.resize(frame, (1310, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)
    if k == ord('a'):
        print('a')
        Xspeed = Xspeed + 1.0

    elif k == ord('d'):
        print('d')
        Xspeed = Xspeed - 1.0
    else:
        pass

    Xaxis.setVelocity(Xspeed)

    if k == 27:
        break

cv2.destroyAllWindows()