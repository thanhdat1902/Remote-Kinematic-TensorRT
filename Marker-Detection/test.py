import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")
# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
results = model.predict(source="p28_front_1.mp4", save=True, save_txt=False)  # save predictions as labels

# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])