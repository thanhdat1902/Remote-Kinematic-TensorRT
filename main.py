from ultralytics import YOLO
import cv2
import numpy as np
import os
from glob import glob
import pandas as pd

def calculate_angle(p1, p2):
    vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    vertical = np.array([0, 1])
    dot_product = np.dot(vector, vertical)
    magnitude = np.linalg.norm(vector) * np.linalg.norm(vertical)
    angle_rad = np.arccos(dot_product / magnitude)
    angle_sign = -np.sign(vector[0])
    angle_deg = angle_sign * np.degrees(angle_rad)
    return angle_deg

class PoseEstimation:
    def __init__(self, image_path, text_path):
        self.model1 = YOLO('yolov8n-pose.pt')
        self.model2 = YOLO('yolov8n-pose-fine-tuned.pt')
        self.model3 = YOLO('yolov8x-pose-p6.pt')
        self.model4 = YOLO('yolov8x-pose-p6-fine-tune.pt')

        self.image_path = image_path
        self.keypoints_model1 = [5, 6, 11, 12]
        self.keypoints_model3 = [5, 6, 11, 12]
        self.text_path = text_path

    def analyze_image(self, yolov8n_pose_angles, yolov8n_pose_fine_tuned_angles, yolov8x_pose_p6_angles, yolov8x_pose_p6_fine_tuned_angles, ground_truth_angles):
        image = cv2.imread(self.image_path)
        if image is None:
            print("Error loading image")
            return

        angle1, angle2, angle3, angle4, angle5 = 0, 0, 0, 0, 0

        # Define colors for each model
        colors = {
            "pretrained 1": (0, 255, 0),
            "finetune 1": (255, 0, 0),
            "Model 3": (255, 0, 0),
            "Model 4": (255, 255, 0),
            "Ground Truth": (255, 0, 255)
        }

        # Process and display results for each model
        results1 = self.model1(image)
        if len(results1) > 0 and results1[0].keypoints is not None:
            keypoints1 = results1[0].keypoints.xy.cpu().numpy()[0]
            angle1 = self.process_keypoints(image, keypoints1, colors["pretrained 1"], "pretrained 1", self.keypoints_model1)
            print("angle1", angle1)

        results2 = self.model2(image)
        if len(results2) > 0 and results2[0].keypoints is not None:
            keypoints2 = results2[0].keypoints.xy.cpu().numpy()[0]
            angle2 = self.process_keypoints_model2(image, keypoints2, colors["finetune 1"], "finetune 1")
            print("angle2", angle2)
        """
        results3 = self.model3(image)
        if len(results3) > 0 and results3[0].keypoints is not None:
            keypoints3 = results3[0].keypoints.xy.cpu().numpy()[0]
            angle3 = self.process_keypoints(image, keypoints3, colors["Model 3"], "Model 3", self.keypoints_model3)
            print("angle3", angle3)

        results4 = self.model4(image)
        if len(results4) > 0 and results4[0].keypoints is not None:
            keypoints4 = results4[0].keypoints.xy.cpu().numpy()[0]
            angle4 = self.process_keypoints_model2(image, keypoints4, colors["Model 4"], "Model 4")
            print("angle4", angle4)
        """
        results5 = self.read_and_process_data(self.text_path)


        height, width = image.shape[:2]
        for i in range(4):
            results5[i][0] *= width
            results5[i][1] *= height
        angle5 = self.process_keypoints_model2(image, results5, colors["Ground Truth"], "Ground Truth")
        print("angle5", angle5)

        # Draw angles on the side of the image
        y_start = 50
        for i, (angle, label) in enumerate(zip([angle1, angle2, angle5], ["pretrained 1", "finetune 1", "Ground Truth"])):
            angleshow = f"{angle:.2f}deg"
            text = f"{label}: {angleshow}"
            font_scale = 1
            thickness = 2
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(image, (10, y_start + i * 30 - h - 5), (10 + w, y_start + i * 30 + 5), (255, 255, 255), -1)
            cv2.putText(image, text, (10, y_start + i * 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors[label],
                        thickness)

        cv2.imshow("Pose Estimation Comparison", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        yolov8n_pose_angles.append(angle1)
        yolov8n_pose_fine_tuned_angles.append(angle2)
        yolov8x_pose_p6_angles.append(angle3)
        yolov8x_pose_p6_fine_tuned_angles.append(angle4)
        ground_truth_angles.append(angle5)

    def process_keypoints(self, image, keypoints, color, label, indices):
        points = {}

        for index in indices:
            if index < len(keypoints):
                x, y = int(keypoints[index, 0]), int(keypoints[index, 1])
                points[index] = (x, y)
                cv2.circle(image, (x, y), 5, color, -1)

        if indices[0] in points and indices[1] in points and indices[2] in points and indices[3] in points:
            midpoint_shoulder = ((points[indices[0]][0] + points[indices[1]][0]) // 2, (points[indices[0]][1] + points[indices[1]][1]) // 2)
            midpoint_hip = ((points[indices[2]][0] + points[indices[3]][0]) // 2, (points[indices[2]][1] + points[indices[3]][1]) // 2)
            cv2.line(image, midpoint_shoulder, midpoint_hip, color, 2)
            angle = calculate_angle(midpoint_shoulder, midpoint_hip)
            #cv2.putText(image, f"{label} Angle: {angle:.2f}°", (midpoint_shoulder[0], midpoint_shoulder[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return angle

    def process_keypoints_model2(self, image, keypoints, color, label):
        if len(keypoints) >= 4:
            shoulder_left = keypoints[0].astype(int)
            shoulder_right = keypoints[1].astype(int)
            mid = keypoints[2].astype(int)
            down = keypoints[3].astype(int)

            midpoint_shoulder = ((shoulder_left[0] + shoulder_right[0]) // 2, (shoulder_left[1] + shoulder_right[1]) // 2)
            midpoint_hip = (down[0], down[1])

            cv2.circle(image, tuple(shoulder_left), 5, color, -1)
            cv2.circle(image, tuple(shoulder_right), 5, color, -1)
            cv2.circle(image, tuple(mid), 5, color, -1)
            cv2.circle(image, tuple(midpoint_hip), 5, color, -1)
            cv2.line(image, midpoint_shoulder, midpoint_hip, color, 2)

            angle = calculate_angle(midpoint_shoulder, midpoint_hip)
            #cv2.putText(image, f"{label} Angle: {angle:.2f}°", (midpoint_shoulder[0], midpoint_shoulder[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return angle

    def read_and_process_data(self, text_path):
        with open(text_path, 'r') as file:
            line = file.readline().strip()

        data = line.split()
        data = data[5:]
        while '2' in data:
            data.remove('2')
        converted_numbers = [float(number) for number in data]
        keypoints = [converted_numbers[i:i + 2] for i in range(0, len(converted_numbers), 2)]
        numpy_array = np.round(keypoints, 2)
        return numpy_array

def run_analyze_image():
    image_dir = r'.\test\images'
    text_dir = r'.\test\labels'

    image_paths = glob(os.path.join(image_dir, '*.png')) + glob(os.path.join(image_dir, '*.PNG'))
    file_pairs = {}

    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        text_filename = base_name.replace('.png', '.txt').replace('.PNG', '.txt')
        text_path = os.path.join(text_dir, text_filename)
        if os.path.exists(text_path):
            file_pairs[image_path] = text_path

    yolov8n_pose_angles = []
    yolov8n_pose_fine_tuned_angles = []
    yolov8x_pose_p6_angles = []
    yolov8x_pose_p6_fine_tuned_angles = []
    ground_truth_angles = []
    i = 0
    for image_path, text_path in file_pairs.items():
        print("image number", i)
        pe = PoseEstimation(image_path, text_path)
        pe.analyze_image(yolov8n_pose_angles, yolov8n_pose_fine_tuned_angles, yolov8x_pose_p6_angles, yolov8x_pose_p6_fine_tuned_angles, ground_truth_angles)
        i += 1
        if i > 10:  # limit to 10 images for demonstration
            break

    print(yolov8n_pose_angles, yolov8n_pose_fine_tuned_angles, yolov8x_pose_p6_angles, yolov8x_pose_p6_fine_tuned_angles, ground_truth_angles)
    data = {
        'yolov8n_pose_angles': yolov8n_pose_angles,
        'yolov8n_pose_fine_tuned_angles': yolov8n_pose_fine_tuned_angles,
        'yolov8x_pose_p6_angles': yolov8x_pose_p6_angles,
        'yolov8x_pose_p6_fine_tuned_angles': yolov8x_pose_p6_fine_tuned_angles,
        'ground_truth_angles': ground_truth_angles
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_file_path = 'angles_data.csv'
    df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    run_analyze_image()
