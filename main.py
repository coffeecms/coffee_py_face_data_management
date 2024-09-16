import os
import cv2
import numpy as np
import json
import base64
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import aiohttp

# Processing mode: 'cpu' or 'gpu'
PROCESSING_MODE = 'cpu'  # Change to 'gpu' for GPU mode

# Load face ID data from file
def load_face_data(filename='data.json'):
    with open(filename, 'r') as f:
        return json.load(f)

# Save face ID data to file
def save_face_data(data, filename='data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Convert image to vector (processed by OpenCV)
def image_to_vector(image, mode='cpu'):
    if mode == 'gpu':
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        gpu_img = cv2.cuda.resize(gpu_img, (128, 128))
        gpu_img_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        return gpu_img_gray.download().flatten()
    else:
        img = cv2.resize(image, (128, 128))  # Resize for consistency
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray.flatten()  # Convert image to vector

# Check if user_id already exists
def check_user_id_exists(user_id, data):
    return any(entry['user_id'] == user_id for entry in data)

# Add face ID data for a worker
def add_face_id(user_id, face_vector, data):
    data.append({'user_id': user_id, 'face_data': face_vector.tolist()})
    save_face_data(data)

# Process an image and add to the system
def process_image(file_path, data, mode):
    user_id = os.path.splitext(os.path.basename(file_path))[0]
    if check_user_id_exists(user_id, data):
        return user_id

    image = cv2.imread(file_path)
    face_vector = image_to_vector(image, mode)

    # Add data to the system
    add_face_id(user_id, face_vector, data)
    return None

# Load face ID data from multiple images in a directory
def load_face_id_from_images(directory, data, mode):
    duplicate_users = []

    # Use ThreadPoolExecutor for concurrent image processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        futures = [executor.submit(process_image, file_path, data, mode) for file_path in file_paths]
        
        for future in futures:
            result = future.result()
            if result:
                duplicate_users.append(result)

    return duplicate_users

# Find the closest face data
def find_closest_face(vector, data):
    closest_user_id = None
    closest_distance = float('inf')

    for entry in data:
        face_vector = np.array(entry['face_data'])
        distance = cosine_similarity([vector], [face_vector])[0][0]
        if distance < closest_distance:
            closest_distance = distance
            closest_user_id = entry['user_id']

    return closest_user_id

# Process a base64 image and find the closest face data
def process_and_search_image(base64_image, data, mode):
    image_data = base64.b64decode(base64_image)
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    face_vector = image_to_vector(image, mode)
    
    closest_user_id = find_closest_face(face_vector, data)
    return closest_user_id

# Handle search requests asynchronously
async def handle_search_request(session, url, base64_image, mode):
    async with session.post(url, json={'base64_image': base64_image, 'mode': mode}) as response:
        return await response.json()

async def main_search(base64_images, mode):
    url = 'http://your-server-endpoint/search'  # Replace with your server endpoint
    async with aiohttp.ClientSession() as session:
        tasks = [handle_search_request(session, url, img, mode) for img in base64_images]
        return await asyncio.gather(*tasks)

# Example usage
if __name__ == "__main__":
    # Load data from data.json
    data = load_face_data()

    # Example base64 images
    base64_images = [
        'base64_image_string_1',
        'base64_image_string_2',
        # Add more base64 image strings
    ]

    # Run concurrent search
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(main_search(base64_images, PROCESSING_MODE))

    # Display results
    for result in results:
        print(result)
