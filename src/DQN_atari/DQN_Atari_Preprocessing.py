import numpy as np
import cv2

def phi_frame(actual_frame,previous_frame):
    max_frame = np.maximum(actual_frame,previous_frame)
    image_gray = cv2.cvtColor(actual_frame,cv2.COLOR_BGR2GRAY)
    resized_im = cv2.resize(image_gray,(84,84),interpolation=cv2.INTER_LINEAR)

    return resized_im

def phi(dqueu,agent_history_length):

    stacked_images = np.zeros(shape=(agent_history_length,84,84))
    for i in range(agent_history_length):
        act_frame = dqueu[i+1]
        prev_frame = dqueu[i]
        image = phi_frame(actual_frame=act_frame,previous_frame=prev_frame)
        stacked_images[i] = image
    return stacked_images