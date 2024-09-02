import json
import os

# Define class mapping
class_mapping = {
    "green_pencil": 0,  # Map "green_pencil" to class ID 0
    "red_pencil":1,
    "mixed_pencil":2
    # Add other classes if necessary
}

# Set paths
json_dir = 'C://Users//Sridhar//Downloads//archive//dataset//train'  # Replace with the path to your JSON files
output_dir = 'C://Users//Sridhar//Downloads//archive//dataset//annotation'  # Replace with the path where you want to save the YOLO annotation files

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through all JSON files in the directory
for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        # Load the JSON annotation file
        with open(os.path.join(json_dir, json_file)) as f:
            annotation_data = json.load(f)

        # Extract image dimensions
        image_width = annotation_data['image_width']
        image_height = annotation_data['image_height']

        # Determine the output file name (same as the image file name but with .txt extension)
        output_file_name = os.path.splitext(json_file)[0] + '.txt'
        output_file_path = os.path.join(output_dir, output_file_name)

        # Open a file to write YOLO annotations
        with open(output_file_path, 'w') as out_file:
            for obj in annotation_data['objects']:
                category = obj['category']
                class_id = class_mapping[category]

                # Extract bounding box
                x_min, y_min, bbox_width, bbox_height = obj['bounding_box']
                
                # Convert to YOLO format
                x_center = (x_min + bbox_width / 2) / image_width
                y_center = (y_min + bbox_height / 2) / image_height
                width_normalized = bbox_width / image_width
                height_normalized = bbox_height / image_height
                
                # Write to YOLO file
                out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_normalized:.6f} {height_normalized:.6f}\n")

        print(f"Annotation file created: {output_file_path}")
