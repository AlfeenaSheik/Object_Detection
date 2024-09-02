import shutil

# Define the original file path and the new file path with .cfg extension
original_file_path = 'yolov3.cfg.txt'
new_file_path = 'yolov3.cfg'

# Rename the file to have a .cfg extension
shutil.move(original_file_path, new_file_path)

new_file_path
