from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Function that takes in user drawing, resizes to shape required by model,
# and scales all intensities to be between 0 and 1
def load_drawing(img):
    img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

## Drawing code, press enter for next drawing, esc to leave ##
drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

# mouse callback function
def interactive_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255),thickness=50)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255),thickness=50)    


img = np.zeros((1024,1024,1), np.uint8)
cv2.namedWindow('Window')
cv2.setMouseCallback('Window',interactive_drawing)
while(1):
    cv2.imshow('Window',img)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
    elif k==13:
        img = load_drawing(img)
        model = load_model('final_model.h5')
        # predict the class
        digit = np.argmax(model.predict(img), axis=-1)
        print(digit[0])
        img = np.zeros((1024,1024,1), np.uint8)
cv2.destroyAllWindows()