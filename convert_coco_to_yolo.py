import json

# Load the COCO file as a dictionary
with open("coco.json", "r") as f:
    coco = json.load(f)

# Open a file for writing the YOLO format annotations
with open("yolo.txt", "w") as f:
    # Iterate over the annotations in the COCO file
    for annotation in coco["annotations"]:
        # Extract the image ID, category ID, and bounding box coordinates from the annotation
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        x_min, y_min, width, height = bbox

        # Convert the category ID to the class ID for YOLO (starting at 0)
        class_id = category_id - 1

        # Convert the bounding box coordinates to the center coordinates and normalized width and height for YOLO
        image_width, image_height = coco["images"][image_id]["width"], coco["images"][image_id]["height"]
        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        width /= image_width
        height /= image_height

        # Write the annotation to the YOLO file in the appropriate format
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
