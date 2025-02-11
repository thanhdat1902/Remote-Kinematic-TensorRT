from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from OKS import OKS

# Load all the models
model_8x = YOLO("yolov8x-pose.pt")
model_8x_ft = YOLO("yolov8x-pose-p6-fine-tune.pt")
model_8n = YOLO("yolov8n-pose.pt")
model_8n_ft = YOLO("yolov8n-pose-fine-tuned.pt")

'''
image = cv2.imread("test/images/frame_0180366.PNG")


results = model_8x(image)

print(results[0].keypoints.xy.cpu().numpy()[0])

for point in results[0].keypoints.xy.cpu().numpy()[0]:
    image = cv2.circle(image, (int(point[0]), int(point[1])), radius=0, color=(0, 0, 255), thickness=10)

cv2.imshow('test', image)
cv2.waitKey(0)
'''

directory = os.fsencode("test/images")

#directory2 = os.fsencode("test/labels")

# helper function to denormalize the ground truth values
def denormalize(list, width, height):
    list[0] = list[0] * width
    list[1] = list[1] * height
    list[2] = list[2] * width
    list[3] = list[3] * height
    list[4] = list[4] * width
    list[5] = list[5] * height
    list[7] = list[7] * width
    list[8] = list[8] * height

    return list

# sigma constant and counter variable
i = 0
sigma = np.array([0.05, 0.05])

df = pd.DataFrame(columns=["FrameID", "yolov8x-pose", "yolov8x-pose-p6-fine-tune", "yolov8n-pose", "yolov8n-pose-fine-tuned"])

# loop through each file in the test images, get the corresponding labels via file name
for file in tqdm(os.listdir(directory)):
    row = []

    # gets the image, labels, and imageID
    imgname = os.fsdecode(file)
    img = cv2.imread(f'test/images/{imgname}')
    labelname = imgname.replace(".PNG", ".txt")
    label = open(f'test/labels/{labelname}', 'r')
    imgid = imgname.replace(".PNG", "")
    imgid = imgid.replace("frame_", "")
    row.append(imgid)

    # read the labels, change to floats, and remove unecessary values
    num_str = label.read()
    nums_list = num_str.split(" ")
    nums_list = list(map(float, nums_list))
    nums_list = nums_list[1:11]

    height, width, _ = img.shape

    #print(width, height)

    # denormalize ground truth values
    nums_list = denormalize(nums_list, width, height)

    # calculate the bounding box area
    bbox_width = nums_list[2]
    bbox_height = nums_list[3]
    area = bbox_width * bbox_height
    #area = 900
    print(nums_list)
    print(area)

    # make a new list containing the ground truth keypoints and visibility flag
    truth = [nums_list[4], nums_list[5], 2, nums_list[7], nums_list[8], 2]
    print(truth)

    # run inference on each model and put the keypoints into a list
    model8x_results = model_8x(f'test/images/{imgname}')[0].keypoints.xy.cpu().numpy()[0]
    model8x_results = [model8x_results[6][0], model8x_results[6][1], 2, model8x_results[5][0], model8x_results[5][1], 2]
    print(model8x_results)

    model8x_ft_results = model_8x_ft(f'test/images/{imgname}')[0].keypoints.xy.cpu().numpy()[0]
    model8x_ft_results = [model8x_ft_results[0][0], model8x_ft_results[0][1], 2, model8x_ft_results[1][0], model8x_ft_results[1][1], 2]
    print(model8x_ft_results)

    model8n_results = model_8n(f'test/images/{imgname}')[0].keypoints.xy.cpu().numpy()[0]
    model8n_results = [model8n_results[6][0], model8n_results[6][1], 2, model8n_results[5][0], model8n_results[5][1], 2]
    print(model8n_results)

    model8n_ft_results = model_8n_ft(f'test/images/{imgname}')[0].keypoints.xy.cpu().numpy()[0]
    model8n_ft_results = [model8n_ft_results[0][0], model8n_ft_results[0][1], 2, model8n_ft_results[1][0], model8n_ft_results[1][1], 2]
    print(model8n_ft_results)

    #calculate OKS for each model
    oks_8x = OKS(truth, model8x_results, sigma, area)
    oks_8x_ft = OKS(truth, model8x_ft_results, sigma, area)
    oks_8n = OKS(truth, model8n_results, sigma, area)
    oks_8n_ft = OKS(truth, model8n_ft_results, sigma, area)

    row.append(oks_8x)
    row.append(oks_8x_ft)
    row.append(oks_8n)
    row.append(oks_8n_ft)

    '''
    print(nums_list)

    img = cv2.rectangle(img, (int(nums_list[0] - nums_list[2] * 0.5), int(nums_list[1] - nums_list[3] * 0.5)), (int(nums_list[0] + nums_list[2] * 0.5), int(nums_list[1] + nums_list[3] * 0.5)), color=(255, 0, 0), thickness=2)
    img = cv2.circle(img, (int(nums_list[4]), int(nums_list[5])), radius=0, color=(0, 0, 255), thickness=10)
    img = cv2.circle(img, (int(nums_list[7]), int(nums_list[8])), radius=0, color=(0, 0, 255), thickness=10)
    cv2.imshow('box test', img)
    cv2.waitKey(0)
    '''    
    
    label.close()

    df.loc[i] = row
    i += 1
    # comment this out if you want to go through the entire dataset
    if i == 5001:
        break

print(f"yolov8x-pose min: {df['yolov8x-pose'].min()}")
print(f"yolov8x-pose max: {df['yolov8x-pose'].max()}")
print(f"yolov8x-pose mean: {df['yolov8x-pose'].mean()}")
print("")
print(f"yolov8x-pose-p6-fine-tune min: {df['yolov8x-pose-p6-fine-tune'].min()}")
print(f"yolov8x-pose-p6-fine-tune max: {df['yolov8x-pose-p6-fine-tune'].max()}")
print(f"yolov8x-pose-p6-fine-tune mean: {df['yolov8x-pose-p6-fine-tune'].mean()}")
print("")
print(f"yolov8n-pose min: {df['yolov8n-pose'].min()}")
print(f"yolov8n-pose max: {df['yolov8n-pose'].max()}")
print(f"yolov8n-pose mean: {df['yolov8n-pose'].mean()}")
print("")
print(f"yolov8n-pose-fine-tuned min: {df['yolov8n-pose-fine-tuned'].min()}")
print(f"yolov8n-pose-fine-tuned max: {df['yolov8n-pose-fine-tuned'].max()}")
print(f"yolov8n-pose-fine-tuned mean: {df['yolov8n-pose-fine-tuned'].mean()}")
print("")

df.to_excel('OKS.xlsx', sheet_name='sheet1')