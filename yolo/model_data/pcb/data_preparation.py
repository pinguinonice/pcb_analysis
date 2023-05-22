import os
import glob
import cv2
import random
import math

def convert_bbox_format(x_center, y_center, h, w, angle):
    angle_rad = math.radians(angle)

    ch = abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad))
    cw = abs(w * math.sin(angle_rad)) + abs(h * math.cos(angle_rad))

    xmin = int(x_center - cw / 2)
    ymin = int(y_center - ch / 2)
    xmax = int(x_center + cw / 2)
    ymax = int(y_center + ch / 2)

    return f"{xmin},{ymin},{xmax},{ymax},0"

def draw_bboxes(image_path, bboxes):
    image = cv2.imread(image_path)
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, _ = map(int, bbox.split(","))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imwrite("annotated_pcb.jpg", image)
    window_name = "Image with bounding boxes"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # Set the desired window size (width, height)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


input_folder = "/Users/philippschneider/Documents/Code/Repositories/pcb_analysis/dataset/pcb_dataset/annotation"
image_folder = "/Users/philippschneider/Documents/Code/Repositories/pcb_analysis/dataset/pcb_dataset/image"
output_file = "./model_data/pcb/train_pcb_all.txt"

bbox_files = glob.glob(os.path.join(input_folder, "*.txt"))

data = []

with open(output_file, "w") as out_f:
    for bbox_file in bbox_files:
        file_name = os.path.splitext(os.path.basename(bbox_file))[0]
        image_path = os.path.join(image_folder, f"{file_name}.jpg") # Change the file extension if necessary

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        bboxes = []
        with open(bbox_file, "r") as bbox_f:
            for line in bbox_f:
                x, y, h, w, angle = map(float, line.strip().split())
                bboxes.append(convert_bbox_format(x, y, h, w, angle))

        output_line = f"{image_path} {' '.join(bboxes)}\n"
        out_f.write(output_line)

        data.append((image_path, bboxes))

# Select a random image and its bounding boxes
random_entry = random.choice(data)
image_path, bboxes = random_entry

# Draw the bounding boxes on the image and display it
#draw_bboxes(image_path, bboxes)

# for alle images in image_folder draw bounding boxes from annotation_test and save in output_folder

input_folder = "/Users/philippschneider/Documents/Code/Repositories/pcb_analysis/dataset/pcb_dataset/annotation_test"
image_folder = "/Users/philippschneider/Documents/Code/Repositories/pcb_analysis/dataset/pcb_dataset/image_test"
output_folder = "model_data/pcb/ground_truth"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

test_bbox_files = glob.glob(os.path.join(input_folder, "*.txt"))

def save_annotated_image(image_path, bboxes, output_path):
    image = cv2.imread(image_path)
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, _ = map(int, bbox.split(","))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 12)

    cv2.imwrite(output_path, image)

for bbox_file in test_bbox_files:
    file_name = os.path.splitext(os.path.basename(bbox_file))[0]
    image_path = os.path.join(image_folder, f"{file_name}.jpg")  # Change the file extension if necessary
    output_path = os.path.join(output_folder, f"{file_name}_annotated.jpg")

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    bboxes = []
    with open(bbox_file, "r") as bbox_f:
        for line in bbox_f:
            x, y, h, w, angle = map(float, line.strip().split())
            bboxes.append(convert_bbox_format(x, y, h, w, angle))

    save_annotated_image(image_path, bboxes, output_path)
