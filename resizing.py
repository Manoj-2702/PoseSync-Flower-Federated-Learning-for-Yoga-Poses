import cv2
import os

def preprocess_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg')):
                # Construct the full file path
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image {img_path}. It may be corrupted or the path is incorrect.")
                    continue
                img_resized = cv2.resize(img, (28, 28))
                
                # Construct the destination path, preserving the subdirectory structure
                rel_path = os.path.relpath(subdir, input_folder)
                dest_path = os.path.join(output_folder, rel_path)
                os.makedirs(dest_path, exist_ok=True)

                # Save the resized image to the destination path
                cv2.imwrite(os.path.join(dest_path, file), img_resized)


input_folder = './data/extractPose/'
output_folder = './data/newImages/'

preprocess_images(input_folder, output_folder)

# input_folder = './data/extractPose/ArdhaChandrasana/'
# output_folder = './data/newImages2/ArdhaChandrasana/'

# preprocess_images(input_folder, output_folder)
