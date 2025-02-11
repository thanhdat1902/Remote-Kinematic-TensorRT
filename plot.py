import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'angles_data_test.csv'
data = pd.read_csv(file_path)
print(f"Number of rows before filtering: {data.shape[0]}")

columns_to_check = [
    'yolov8n_pose_angles',
    'yolov8n_pose_fine_tuned_angles',
    'yolov8x_pose_p6_angles',
    'yolov8x_pose_p6_fine_tuned_angles',
    'ground_truth_angles'
]

data = data[~data[columns_to_check].gt(90).any(axis=1)]
print(f"Number of rows after filtering: {data.shape[0]}")

yolov8n_pose_angles = data['yolov8n_pose_angles']
yolov8n_pose_fine_tuned_angles = data['yolov8n_pose_fine_tuned_angles']
yolov8x_pose_p6_angles = data['yolov8x_pose_p6_angles']
yolov8x_pose_p6_fine_tuned_angles = data['yolov8x_pose_p6_fine_tuned_angles']
ground_truth_angles = data['ground_truth_angles']
frame_ids = range(len(ground_truth_angles))
plt.rcParams.update({'font.size': 28})

# Plotting angles with dots
plt.figure(figsize=(20, 10))
plt.scatter(frame_ids, yolov8n_pose_angles, label='yolov8n_pose_angles', color='green', s=10)
plt.scatter(frame_ids, yolov8n_pose_fine_tuned_angles, label='yolov8n_pose_fine_tuned_angles', color='red', s=10)
plt.scatter(frame_ids, yolov8x_pose_p6_angles, label='yolov8x_pose_p6_angles', color='blue', s=10)
plt.scatter(frame_ids, yolov8x_pose_p6_fine_tuned_angles, label='yolov8x_pose_p6_fine_tuned_angles', color='yellow', s=10)
plt.scatter(frame_ids, ground_truth_angles, label='ground_truth_angles', color='purple', s=10)

plt.xlabel('Frame ID')
plt.ylabel('Trunk Lateral Flexion Angle (degrees)')
plt.title('Comparison of Trunk Lateral Flexion Angles')
plt.legend()
plt.show()

# Calculate errors
error_yolov8n = np.abs(np.array(yolov8n_pose_angles) - np.array(ground_truth_angles))
error_yolov8n_finetuned = np.abs(np.array(yolov8n_pose_fine_tuned_angles) - np.array(ground_truth_angles))
error_yolov8x_p6 = np.abs(np.array(yolov8x_pose_p6_angles) - np.array(ground_truth_angles))
error_yolov8x_p6_finetuned = np.abs(np.array(yolov8x_pose_p6_fine_tuned_angles) - np.array(ground_truth_angles))

# Plotting errors with dots
plt.figure(figsize=(20, 10))
plt.scatter(frame_ids, error_yolov8n, label='Error yolov8n_pose_angles', color='green', s=10)
plt.scatter(frame_ids, error_yolov8n_finetuned, label='Error yolov8n_pose_fine_tuned_angles', color='red', s=10)
plt.scatter(frame_ids, error_yolov8x_p6, label='Error yolov8x_pose_p6_angles', color='blue', s=10)
plt.scatter(frame_ids, error_yolov8x_p6_finetuned, label='Error yolov8x_pose_p6_fine_tuned_angles', color='yellow', s=10)

plt.xlabel('Frame ID')
plt.ylabel('Absolute Error in Trunk Lateral Flexion Angle (degrees)')
plt.title('Error Comparison of Trunk Lateral Flexion Angles')
plt.legend()
plt.show()

# Calculate percentage errors
percentage_error_yolov8n = (error_yolov8n / np.abs(ground_truth_angles)) * 100
percentage_error_yolov8n_finetuned = (error_yolov8n_finetuned / np.abs(ground_truth_angles)) * 100
percentage_error_yolov8x_p6 = (error_yolov8x_p6 / np.abs(ground_truth_angles)) * 100
percentage_error_yolov8x_p6_finetuned = (error_yolov8x_p6_finetuned / np.abs(ground_truth_angles)) * 100

# Plotting percentage errors with dots
plt.figure(figsize=(20, 10))
plt.scatter(frame_ids, percentage_error_yolov8n, label='Percentage Error yolov8n_pose_angles', color='green', s=10)
plt.scatter(frame_ids, percentage_error_yolov8n_finetuned, label='Percentage Error yolov8n_pose_fine_tuned_angles', color='red', s=10)
plt.scatter(frame_ids, percentage_error_yolov8x_p6, label='Percentage Error yolov8x_pose_p6_angles', color='blue', s=10)
plt.scatter(frame_ids, percentage_error_yolov8x_p6_finetuned, label='Percentage Error yolov8x_pose_p6_fine_tuned_angles', color='yellow', s=10)

plt.xlabel('Frame ID')
plt.ylabel('Percentage Error in Trunk Lateral Flexion Angle (%)')
plt.title('Percentage Error Comparison of Trunk Lateral Flexion Angles')
plt.legend()
plt.show()
