import os
import shutil
import random

# Paths to your image and annotation folders
image_dir = 'C://Users//Sridhar//Downloads//archive//dataset//train'  # Update to use forward slashes
annotation_dir = 'C://Users//Sridhar//Downloads//archive//dataset//annotation'  # Update to use forward slashes

# Paths to the train and val folders
train_image_dir = 'C://Users//Sridhar//Downloads//yolo//dataset//images//train'
val_image_dir = 'C://Users//Sridhar//Downloads//yolo//dataset//images//val'
train_label_dir = 'C://Users//Sridhar//Downloads//yolo//dataset//labels//train'
val_label_dir = 'C://Users//Sridhar//Downloads//yolo//dataset//labels//val'

# Ensure the output directories exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get a list of all images
images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(images)} images in {image_dir}")

# Shuffle the images list
random.shuffle(images)

# Set your split ratio (e.g., 80% train, 20% val)
split_ratio = 0.8
train_count = int(len(images) * split_ratio)

# Split the images into train and val sets
train_images = images[:train_count]
val_images = images[train_count:]
print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")

# Function to move files
def move_files(image_list, target_image_dir, target_label_dir):
    for image in image_list:
        # Move image file
        image_path = os.path.join(image_dir, image)
        target_image_path = os.path.join(target_image_dir, image)
        if os.path.exists(image_path):
            shutil.copy(image_path, target_image_path)
            print(f"Copied image: {image_path} to {target_image_path}")
        else:
            print(f"Image file not found: {image_path}")

        # Move corresponding annotation file
        annotation_file = os.path.splitext(image)[0] + '.txt'
        annotation_path = os.path.join(annotation_dir, annotation_file)
        target_annotation_path = os.path.join(target_label_dir, annotation_file)
        if os.path.exists(annotation_path):
            shutil.copy(annotation_path, target_annotation_path)
            print(f"Copied annotation: {annotation_path} to {target_annotation_path}")
        else:
            print(f"Annotation file not found: {annotation_path}")

# Move training files
move_files(train_images, train_image_dir, train_label_dir)

# Move validation files
move_files(val_images, val_image_dir, val_label_dir)

print(f"Dataset successfully split: {len(train_images)} training and {len(val_images)} validation images.")
