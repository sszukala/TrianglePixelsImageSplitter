!pip install Pillow
!pip install requests
!pip install matplotlib
!pip install torch torchvision torchaudio

import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def download_image(url: str) -> bytes:
    """Download an image from the given URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.content
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def split_pixels(image_array: np.ndarray) -> np.ndarray:
    """Split square pixels into triangles."""
    rows, cols = image_array.shape
    triangle_pixel_array = np.zeros((rows * 2, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols // 2):
            triangle_pixel_array[i * 2, j * 2 : (j + 1) * 2] = image_array[i, j]
            triangle_pixel_array[(i * 2) + 1, j * 2 : (j + 1) * 2] = image_array[i, j]
    return triangle_pixel_array

def convert_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert the given image array into a PyTorch tensor."""
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

class TriangleSplitter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split square pixels into triangles in the given tensor."""
        # Reshape the input tensor (assuming input is [batch, channels, height, width])
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3)) 

        # Create an empty tensor to store the split pixels with the correct size
        split_pixels = torch.zeros(x.size(0), x.size(1), x.size(2) * 2, x.size(3), device=x.device) 

        # Split each square pixel into two triangles
        # Updated slicing to match the intended logic and the shape of split_pixels
        for i in range(x.size(2)):
            split_pixels[:, :, i * 2, :] = x[:, :, i, :]  # Copy row i to row i*2
            split_pixels[:, :, i * 2 + 1, :] = x[:, :, i, :]  # Copy row i to row i*2 + 1

        return split_pixels

class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.triangle_splitter = TriangleSplitter()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.triangle_splitter(x) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

def display_feature_maps(feature_maps: torch.Tensor):
    """Display the extracted feature maps."""
    num_maps = feature_maps.shape[1]
    fig, axes = plt.subplots(1, num_maps, figsize=(num_maps * 4, 4))

    for i in range(num_maps):
        axes[i].imshow(feature_maps[0, i].detach().cpu().numpy(), cmap='gray')
        axes[i].set_title(f'Feature Map {i + 1}')
        axes[i].axis('off')

    plt.show()

def display_images_side_by_side(original_image: np.ndarray, converted_image: np.ndarray):
    """Display the original and converted images side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  

    axes[0].imshow(original_image, cmap='gray', aspect='auto')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(converted_image, cmap='gray', aspect='auto')
    axes[1].set_title('Converted Image (Split Pixels)')
    axes[1].axis('off')

    plt.show()

def main():
    image_url = "https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg"

    content = download_image(image_url)

    if content is not None:
        with BytesIO(content) as bio:
            original_image = Image.open(bio).convert('L')
            image_array = np.array(original_image)
            triangle_pixel_array = split_pixels(image_array)
            converted_image = Image.fromarray(triangle_pixel_array)

            display_images_side_by_side(image_array, triangle_pixel_array)

            image_tensor = convert_image_to_tensor(triangle_pixel_array)

            cnn = FeatureExtractorCNN()
            feature_maps = cnn(image_tensor)

            display_feature_maps(feature_maps)

            converted_image.save('triangle_pixels_dog.jpg')
            print(f"Converted image saved to: triangle_pixels_dog.jpg")
    else:
        print("Skipping image processing due to download error.")

if __name__ == '__main__':
    main()
