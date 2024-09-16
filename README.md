# Face ID Management System - https://blog.lowlevelforest.com/
 
## Overview

This system manages Face IDs for workers, allowing you to load face data from images, search for the closest face data based on a base64 image, and handle concurrent searches. It supports both CPU and GPU processing modes.

## Features

- Load face ID data from images.
- Search for the closest face data based on a base64 image.
- Handle concurrent searches efficiently.
- Support both CPU and GPU processing modes.

## Requirements

- Python 3.8+
- OpenCV with CUDA (for GPU mode)
- NumPy
- scikit-learn
- aiohttp (for asynchronous requests)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-repo/face-id-management.git
    cd face-id-management
    ```

2. **Install dependencies:**

    Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

    Install required Python libraries:

    ```bash
    pip install opencv-python-headless numpy scikit-learn aiohttp
    ```

    To use GPU processing mode, ensure OpenCV is built with CUDA support. You might need to build OpenCV from source or find a pre-built version with CUDA support.

## Configuration

- **Processing Mode:**
  Set `PROCESSING_MODE` to `'cpu'` or `'gpu'` in the script to choose the processing mode.

- **Server Endpoint:**
  Update the `url` variable in the `main_search` function to your server endpoint for handling search requests.

## Usage

1. **Loading Face Data from Images:**

    Place images in a directory and call the `load_face_id_from_images` function:

    ```python
    directory = 'path/to/your/image/folder'
    data = load_face_data()
    duplicates = load_face_id_from_images(directory, data, PROCESSING_MODE)
    ```

    This function will process all images in the specified directory and add face data to `data.json`. It also checks for duplicate `user_id`s.

2. **Searching for Face Data:**

    Convert base64 image strings to vectors and find the closest face data:

    ```python
    base64_images = [
        'base64_image_string_1',
        'base64_image_string_2',
        # Add more base64 image strings
    ]

    results = loop.run_until_complete(main_search(base64_images, PROCESSING_MODE))
    ```

    This will perform concurrent searches for all provided base64 images.

3. **Example Base64 Image Conversion:**

    To convert an image to a base64 string:

    ```python
    import base64

    with open('path/to/image.jpg', 'rb') as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    ```

## Running the System

To run the system and test the functionalities:

```bash
python your_script.py
```

Ensure you replace `your_script.py` with the actual script name.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
