import os
import shutil
import random
import zipfile

# Paths to the dataset
dataset_zip_path = 'C:/Users/Sridhar/Downloads/archive.zip'  # Path to the uploaded ZIP file
unzip_dir = 'C:/Users/Sridhar/Downloads/yolo/dataset'  

# Unzip the dataset
with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)

# Paths to images and annotations within the unzipped folder
image_dir = os.path.join(unzip_dir, 'images')  # Assuming images are in an 'images' folder
annotation_dir = os.path.join(unzip_dir, 'annotations')  # Assuming annotations are in an 'annotations' folder

# Paths to train and validation folders
train_image_dir = os.path.join(unzip_dir, 'train_images')
val_image_dir = os.path.join(unzip_dir, 'val_images')
train_label_dir = os.path.join(unzip_dir, 'train_labels')
val_label_dir = os.path.join(unzip_dir, 'val_labels')

# Ensure the output directories exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get a list of all images
images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle the images list
random.shuffle(images)

# Set your split ratio (e.g., 80% train, 20% val)
split_ratio = 0.8
train_count = int(len(images) * split_ratio)

# Split the images into train and val sets
train_images = images[:train_count]
val_images = images[train_count:]

# Function to move files
def move_files(image_list, target_image_dir, target_label_dir):
    for image in image_list:
        shutil.copy(os.path.join(image_dir, image), os.path.join(target_image_dir, image))
        
        annotation_file = os.path.splitext(image)[0] + '.txt'
        shutil.copy(os.path.join(annotation_dir, annotation_file), os.path.join(target_label_dir, annotation_file))

# Move training and validation files
move_files(train_images, train_image_dir, train_label_dir)
move_files(val_images, val_image_dir, val_label_dir)

# Create the .data file
def create_data_file(classes_count, train_txt_path, val_txt_path, names_path, backup_path, output_path):
    with open(output_path, 'w') as f:
        f.write(f"classes= {classes_count}\n")
        f.write(f"train = {train_txt_path}\n")
        f.write(f"valid = {val_txt_path}\n")
        f.write(f"names = {names_path}\n")
        f.write(f"backup = {backup_path}\n")

# Assuming a single class 'pencil'
classes_count = 1
train_txt_path = os.path.join(unzip_dir, 'train.txt')
val_txt_path = os.path.join(unzip_dir, 'val.txt')
names_path = os.path.join(unzip_dir, 'obj.names')
backup_path = 'backup/'
output_data_path = os.path.join(unzip_dir, 'obj.data')

create_data_file(classes_count, train_txt_path, val_txt_path, names_path, backup_path, output_data_path)

# Create the .names file
def create_names_file(class_names, output_path):
    with open(output_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

create_names_file(['pencil'], names_path)

# Update the .cfg file
def update_cfg_file(cfg_path, classes_count, output_path):
    with open(cfg_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith('filters='):
            lines[i] = f"filters={3 * (classes_count + 5)}\n"
        elif line.startswith('classes='):
            lines[i] = f"classes={classes_count}\n"

    with open(output_path, 'w') as f:
        f.writelines(lines)

cfg_path = 'yolov3.cfg'  # Replace with your base .cfg file path
output_cfg_path = os.path.join(unzip_dir, 'yolov3_custom.cfg')

update_cfg_file(cfg_path, classes_count, output_cfg_path)

# Create train.txt and val.txt files
def create_image_paths_file(image_list, image_dir, output_path):
    with open(output_path, 'w') as f:
        for image in image_list:
            f.write(f"{os.path.join(image_dir, image)}\n")

create_image_paths_file(train_images, train_image_dir, train_txt_path)
create_image_paths_file(val_images, val_image_dir, val_txt_path)

# Output paths for reference
print(f"Training images and labels are in: {train_image_dir}")
print(f"Validation images and labels are in: {val_image_dir}")
print(f".data file created at: {output_data_path}")
print(f".names file created at: {names_path}")
print(f"Custom .cfg file created at: {output_cfg_path}")
print(f"train.txt created at: {train_txt_path}")
print(f"val.txt created at: {val_txt_path}")

# Instruction for running the training
print("\nTo start training, run the following command:")
print(f"./darknet detector train {output_data_path} {output_cfg_path} path/to/darknet53.conv.74")
