! pip install timm transformers googledrivedownloader opencv-python matplotlib scikit-image

import cv2
import csv
import time
import requests
import os
import torch
import requests
import json
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import defaultdict
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from google_drive_downloader import GoogleDriveDownloader as gdd


torch.set_grad_enabled(False);
# MDETR
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# standard PyTorch mean-std input image normalization

gdd.download_file_from_google_drive(file_id='11BlSLeDE1dpa5Sm71ef4jYJ0OtPSqSEk',
                                    dest_path='content/AVA_Metrics/AVA_Metrics.zip',
                                    unzip=True)
# download binaries to be used in the metrics

with open('content/AVA_Metrics/ava_v2.2/ava_val_v2.2.csv', mode='r+') as csv_file:
  with open('content/video_ID.csv', mode='w') as csv_test:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for_comparison = 0
    for row in csv_reader:
        if row[0] != for_comparison:
          csv_test.write(f"{row[0]}\n")
          line_count += 1
          for_comparison = row[0]
    print(f'Processed {line_count} videos.')
# Because of not having the URL to download video_IDs directly from AVA webpage, it creates this file from the AVA_validation file .

open("MDETR_results.csv", "w")
# Create file for results.

!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1m6jU4EI-3wgzSg_A-0OnDUh-clgnYUxo' -O content/ava_train_v2.2.csv
# Download CSV to compare

!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nNUm7EDnd24Y95CE2oxC-LwcQimljKgR' -O content/questions_list.txt
# txt file that saves questions

"""## *Define model*"""

model_qa = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5_gqa', pretrained=True, return_postprocessor=False)
model_qa = model_qa.cuda()
model_qa.eval()

# We download the mapping from the answers to their id.
answer2id_by_type = json.load(requests.get("https://nyu.box.com/shared/static/j4rnpo8ixn6v0iznno2pim6ffj3jyaj8.json", stream=True).raw)
id2answerbytype = {}                                                       
for ans_type in answer2id_by_type.keys():                        
    curr_reversed_dict = {v: k for k, v in answer2id_by_type[ans_type].items()}
    id2answerbytype[ans_type] = curr_reversed_dict

"""## *Process*

### Boxes and plots
"""

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, scores, boxes, masks=None):
    plt.figure(figsize=(16,10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(masks)
    for s, (xmin, ymin, xmax, ymax), mask, c in zip(scores, boxes.tolist(), masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))

        if mask is None:
          continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=c)
          ax.add_patch(p)


    plt.imshow(np_image)
    plt.axis('off')
    plt.show()

def plot_image(pil_img):
    plt.figure(figsize=(16,10))
    np_image = np.array(pil_img)
    plt.imshow(np_image)
    plt.axis('off')
    plt.show()

"""### Video Processing"""

def get_data_from_video(im, caption):
  # mean-std normalize the input image (batch-size: 1)
  img = transform(im).unsqueeze(0).cuda()

  # propagate through the model
  memory_cache = model_qa(img, [caption], encode_and_save=True)
  outputs = model_qa(img, [caption], encode_and_save=False, memory_cache=memory_cache)

  # keep only predictions with 0.7+ confidence
  probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
  keep = (probas > CONFIDENCE_TOL).cpu()

  # Classify the question type
  type_conf, type_pred = outputs["pred_answer_type"].softmax(-1).max(-1)
  ans_type = type_pred.item()
  types = ["obj", "attr", "rel", "global", "cat"]

  ans_conf, ans = outputs[f"pred_answer_{types[ans_type]}"][0].softmax(-1).max(-1)
  answer = id2answerbytype[f"answer_{types[ans_type]}"][ans.item()]

  if answer == "yes":
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)
    # plot_results(im, probas[keep], bboxes_scaled)

    return box_cxcywh_to_xyxy(outputs['pred_boxes'].cpu()[0, keep]).tolist()
    # MDETR returns x and y initial, and width and height, so we convert
    # it to initial x and y, and end x and y

  return None

def define_question(actions_filename, action_id=None):
  """
  Defines which question use to search for the correct
  action, given the action ID, using a questions list file.
  """
  with open(actions_filename) as file_handler:
    file_content = file_handler.read()

  file_content = file_content.split("\n")

  if action_id != None:
    question = file_content[action_id - 1]
    # Actions ID start with 1, list with 0

    return question

  # If action_id is defined, return the questions asked
  # Else, returns questions list

  return file_content

def append_boxes_frames(im, captions):
  frames = list()
  action_ids = list()

  for caption in captions:
    boxes_axes = get_data_from_video(im, caption)

    if boxes_axes != None:
      for box in boxes_axes:
        if len(box) > 1:
          frames.append(box)
          action_ids.append(captions.index(caption))

  return frames, action_ids

def get_n_frames(video):
  return int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  # Metadata get from OpenCV

def process_video(path, captions):
  global succeed_frames
  
  capture = cv2.VideoCapture(path)
  succeed_frames = list()

  if isinstance(get_n_frames(capture), int):
  # Check if the video's url has frames verifying that they are valid

    FPS = round(capture.get(cv2.CAP_PROP_FPS)) or 30
    # In this way, we use video FPS, or 30 if get video FPS fails
    for frame_index in range(902, 1798, 1):
      # Step must be number of fps
      capture.set(1, frame_index * FPS + int(FPS / 2))
      # AVA uses the middle frame
      success, frame = capture.read()

      if success:
        im = Image.fromarray(frame)

        boxes, actions = append_boxes_frames(im, captions)

        for box, action in zip(boxes, actions):
          succeed_frames.append(
              [
                  int(frame_index), # Data saved in seconds from start
                  *box, # Unpack frame coords
                  action, # ACTION ID for each frame
                  boxes.index(box) # In the same frame, box number get in order
              ]
          )
        
        print('frame',frame_index, 'of 1798.', end='\r')

  capture.release()

  return succeed_frames

def write_csv(name, video_data):
  """
  Receives video content and writes it in a CSV file.
  """
  with open("MDETR_results.csv", "a") as file_handler:
      for video_data_list in video_data:
          new_list = [name, *video_data_list]
          text_line = "{},{:04d},{:.3f},{:.3f},{:.3f},{:.3f},{},{}\n".format(*new_list)
          file_handler.write(text_line)
          # Parameters written
          # 1. Video name
          # 2. Time of event
          # 3. x of first extreme of rectangle
          # 4. y of first extreme of rectangle
          # 5. x of second extreme of rectangle
          # 6. y of second extreme of rectangle
          # 7. Action ID
          # 8. Index to recognize rects if there are more than one in an image

def iter_videos(flag, N_VIDEOS_TEST):
  """
  Gets a list of video names in an url and analyze each one. Deletes
  not in use files to save space.
  """
  format_videos = [".mp4", ".mkv", ".webm"]

  questions = define_question("content/questions_list.txt")

  with open('content/video_ID.csv', mode='r') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      line_count = 0

      for row in csv_reader:
         for format_video in format_videos:
            name = row[0]
            url = "https://s3.amazonaws.com/ava-dataset/trainval/"
            try:
                video_data = process_video(url + name + format_video, questions)
                # Returns list of frames that answer question
                write_csv(name, video_data)

            except Exception as e:
                print(e)

         line_count += 1 
         if (flag==False) and (line_count >= N_VIDEOS_TEST):
                break

"""## *Main Sets*"""

CONFIDENCE_TOL = 0.7
N_VIDEOS_TEST = 1
# With this var we select number of videos tested.

os.remove('MDETR_results.csv')
# Delete previous file to not append results. Use it only if the test runs complete.
iter_videos(False, N_VIDEOS_TEST)
# 'True' if you want to analize all videos.
# 'False' if you want to analize N_VIDEOS.

"""# **Metrics**"""

!python -O content/AVA_Metrics/calc_mAP.py -l content/AVA_Metrics/ava_v2.2/ava_action_list_v2.2_for_activitynet_2019.pbtxt  -g content/AVA_Metrics/ava_v2.2/ava_val_v2.2.csv -e content/AVA_Metrics/ava_v2.2/ava_val_excluded_timestamps_v2.2.csv -d MDETR_results.csv