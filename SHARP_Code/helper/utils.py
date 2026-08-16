import os
from PIL import Image
from io import BytesIO
import base64
import json
import csv
import re

from typing import Union

def image_to_bytes(value: Union[str,Image.Image]) -> bytes:
    if isinstance(value, Image.Image):
        buffer = BytesIO()
        value = value.convert('RGB')
        value.save(buffer, format='JPEG')
        img_buffer = buffer.getvalue()
    elif isinstance(value, str):
        assert os.path.exists(value), f'image file not found: {value}'
        """Getting the base64 string"""
        with open(value, "rb") as image_file:
            img_buffer = image_file.read()
    else:
        print('invalid data, return None')
        return None
    return img_buffer

def encode_image_base64(value: Union[str,Image.Image, bytes]) -> str:
    if isinstance(value, bytes):
        img_buffer = value
    else:
        img_buffer = image_to_bytes(value)
    if img_buffer is None:
        return None
    """Encoding the image to base64 string"""
    return base64.b64encode(img_buffer).decode("utf-8")
    
def decode_image_base64(b64_string: str) -> Image.Image:
    """Decodes a base64 string to an image."""
    img_data = base64.b64decode(b64_string)
    image = Image.open(BytesIO(img_data))
    return image.convert('RGB')  # Ensure the image is in RGB format

# load samples
def load_csv(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list[dict]: A list of dictionaries representing the rows in the CSV file.
    """
    data = []
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def extract_json_from_string(input_string):
    """
    Converts a string containing a JSON object into a Python dictionary.

    This function handles strings where the JSON is either embedded in
    markdown-style code blocks or appended to other text.

    Args:
        input_string: The input string containing the JSON data.

    Returns:
        A dictionary representing the JSON object.
    """
    try:
        # Case 1: JSON is within ```json ... ```
        match = re.search(r'```json\n(.*?)\n```', input_string, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)

        # Case 2: JSON is at the end of the string
        start_index = input_string.find('{')
        if start_index != -1:
            json_str = input_string[start_index:]
            return json.loads(json_str)

        # If no specific format is matched, try to load the whole string
        return json.loads(input_string)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None




