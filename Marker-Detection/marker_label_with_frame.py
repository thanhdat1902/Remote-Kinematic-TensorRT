import os
folder_name = "P15"
newFile = 1001

ori = 0


current_path = os.path.join(os.getcwd(), f"Label/val/{folder_name}/obj_train_data")

for filename in os.listdir(f"./Label/val/{folder_name}/obj_train_data"):
    old_file_path = os.path.join(current_path, "frame_{:06d}.txt".format(ori))
    new_file_path = os.path.join(current_path, "frame_{:06d}.txt".format(newFile))
    os.rename(old_file_path, new_file_path)
    ori+=1
    newFile+=1