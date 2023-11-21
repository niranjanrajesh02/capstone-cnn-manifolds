import cv2
import argparse 
import os
from data_utils import center_crop, central_resize
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='mug_test', help='video file name')
parser.add_argument('--size', type=str, default=299, help='size of frame')
args = parser.parse_args()

video_path = './videos/' + args.video + '.mp4'
save_path = './frames/' + args.video


if os.path.exists(save_path):
    # delete the folder and its contents
    shutil.rmtree(save_path)


os.mkdir(save_path)


vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()

count = 0
while success:
  # image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_AREA)
  image = central_resize(image, (int(args.size), int(args.size)))
  frame_path = f'{save_path}/frame_{count}.jpg'
  cv2.imwrite(frame_path, image)        
  print('Read + Saved a new frame: ', success)
  success,image = vidcap.read()
  count += 1
  if count == 10:
    break