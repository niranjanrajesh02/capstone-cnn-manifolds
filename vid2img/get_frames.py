import cv2
import argparse
import os
from data_utils import center_crop, central_resize
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='mug',
                    help='video file name')  # sunglass or mug
parser.add_argument('--size', type=str, default=299, help='size of frame')
parser.add_argument('--env', type=str, default='pc', help='environment')
args = parser.parse_args()

if args.env == 'pc':
    video_path = './videos/' + args.video + '/'
    save_path = './frames/' + args.video + '/'
elif args.env == 'hpc':
    video_path = '/home/niranjan.rajesh_asp24/capstone-cnn-manifolds/vid2img/videos/' + args.video + '/'
    save_path = '/home/niranjan.rajesh_asp24/capstone-cnn-manifolds/vid2img/frames/' + args.video + '/'



if os.path.exists(save_path):
    # delete the folder and its contents
    shutil.rmtree(save_path)


os.mkdir(save_path)

videos = os.listdir(video_path)

count = 0
for video in videos:
    video_path = video_path + video
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    while success:
        # image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_AREA)
        image = central_resize(image, (int(args.size), int(args.size)))
        frame_path = f'{save_path}/{args.video}_frame_{count}.jpg'
        cv2.imwrite(frame_path, image)
        print(f'Read + Saved a new frame {count}')
        success, image = vidcap.read()
        count += 1
        # if count == 20:
        #     break
    print(f"Successfully read and saved {count} {args.video} frames")
