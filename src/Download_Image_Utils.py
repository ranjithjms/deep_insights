import requests
import os
import json
from pycocotools.coco import COCO
from PIL import Image
from io import BytesIO

def download_annotations(url, save_path):
    """Download COCO annotations file."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Annotations file saved to {save_path}")
    else:
        print("Failed to download annotations file.")

def download_coco_image(image_id, annotations_file, save_dir='images'):
    """Download image from COCO dataset using image ID."""
    # Load COCO annotations
    coco = COCO(annotations_file)

    # Fetch image metadata using pycocotools
    img_info = coco.loadImgs(image_id)[0]
    image_url = img_info['coco_url']

    # Download the image
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_path = os.path.join(save_dir, f"{img_info['file_name']}")
        image.save(image_path)
        print(f"Image saved to {image_path}")
        return image_path
    else:
        print("Failed to download image.")
        return None

# URLs
annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
annotations_file = 'annotations/instances_train2017.json'

# Download annotations file if not already present
if not os.path.exists(annotations_file):
    download_annotations(annotations_url, 'annotations_trainval2017.zip')
    # Extract the zip file
    import zipfile
    with zipfile.ZipFile('annotations_trainval2017.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove('annotations_trainval2017.zip')

# Example usage
image_id = 391895  # You can change this to any valid image ID from the COCO dataset
downloaded_image_path = download_coco_image(image_id, annotations_file)
