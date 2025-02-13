# TrianglePixelsImageSplitter
This code downloads an image, converts it to grayscale, splits the image pixels into triangles using a novel technique, and then applies a CNN for feature extraction. 

# Image Feature Extraction with Triangle Pixel Splitting

This project demonstrates a technique for extracting features from images using a Convolutional Neural Network (CNN) after splitting the image pixels into triangles. This approach potentially offers a unique way to represent image information, which could be explored for various computer vision tasks.

## How it Works

1. **Image Download:** An image is downloaded from a given URL using the `requests` library.
2. **Grayscale Conversion:** The image is converted to grayscale using the `PIL` library.
3. **Triangle Pixel Splitting:** The grayscale image is then processed to split each square pixel into two triangles. This is done by duplicating each pixel row and shifting pixel values.
4. **CNN Feature Extraction:** The triangle-split image is fed into a CNN to extract features. The CNN consists of convolutional layers, ReLU activation, and max pooling layers.
5. **Feature Map Visualization:** The extracted feature maps from the CNN are displayed using `matplotlib`.
6. **Image Display:** The original and converted (triangle-split) images are displayed side-by-side for comparison.
7. **Image Saving:** The converted image is saved to a file.

## Usage

1. **Install Dependencies:** Make sure you have the required libraries installed:

2.  **Run the Code:** Execute the Python script (`main.py`).
3. **View Results:** The output will include:
    - Side-by-side display of the original and converted images.
    - Visualization of the extracted feature maps.
    - A saved copy of the converted image.


## Code Structure

- **`download_image(url)`:** Downloads the image from the given URL.
- **`split_pixels(image_array)`:** Splits the image pixels into triangles.
- **`convert_image_to_tensor(image)`:** Converts the image array to a PyTorch tensor.
- **`TriangleSplitter`:** A PyTorch module that performs triangle pixel splitting within the CNN.
- **`FeatureExtractorCNN`:** The CNN model for feature extraction.
- **`display_feature_maps(feature_maps)`:** Displays the feature maps.
- **`display_images_side_by_side(original_image, converted_image)`:** Displays the original and converted images.
- **`main()`:** The main function that orchestrates the process.


## Potential Applications

This technique could be investigated for applications like:

- **Image Recognition:** Exploring whether the triangle pixel representation provides any advantages for classification tasks.
- **Image Segmentation:** Applying the CNN to the triangle-split image for segmenting objects.
- **Texture Analysis:** Analyzing the impact of triangle splitting on texture features.
- **Image Compression:** Evaluating the possibility of using triangle-split representations for compression.

## Limitations

- The current implementation uses a simple triangle pixel splitting method. More sophisticated techniques might be explored.
- The CNN architecture is basic. Deeper and more specialized networks could be used.


## Contributing

Feel free to contribute to this project by:

- Suggesting improvements to the triangle pixel splitting or CNN architecture.
- Experimenting with different image datasets and applications.
- Adding more visualization or analysis features.

## License

MIT

Hypothesized Benefits

Enhanced Edge and Texture Representation: By splitting square pixels into triangles, the code introduces a form of implicit edge detection. The boundaries between triangles can potentially highlight edges and fine textures in the image more effectively than the original square pixel grid. This could make it easier for the CNN to learn features related to edges, textures, and object boundaries.

Increased Feature Diversity: The triangle representation potentially creates a more diverse set of local patterns for the CNN to learn from. Instead of just relying on square pixel neighborhoods, the CNN can now extract features based on the relationships between triangles with different orientations. This increased feature diversity could lead to a richer and more robust feature representation of the image.

Rotation Invariance (Potentially): While the code doesn't explicitly address rotation invariance, the use of triangles might inherently provide some level of resistance to small rotations. Since triangles have different orientations, features learned from them might be less sensitive to slight changes in image rotation compared to features extracted from square pixels.

Improved Feature Extraction in CNNs: CNNs excel at extracting local patterns and hierarchies of features. By introducing triangular patterns, the code potentially creates new and potentially more informative local structures that the CNN can leverage to build its feature representations. This could lead to more accurate and discriminative features for tasks like image classification or object detection.

Important Considerations

Empirical Validation: It's crucial to note that these benefits are largely hypothetical. Whether the triangle pixel splitting actually improves image quality and feature extraction would need to be rigorously evaluated through experiments and comparisons with standard CNN approaches.
Task Dependence: The effectiveness of this technique might vary significantly depending on the specific computer vision task. For certain tasks, it might provide an advantage, while for others, it might not make a significant difference or could even be detrimental.
Computational Cost: The triangle splitting process adds computational overhead. This cost needs to be considered in relation to the potential benefits gained.
Alternative Approaches: There are other established methods for enhancing edge and texture representation in images, such as using specialized filters or pre-processing techniques. A thorough comparison with these alternatives would be necessary to determine the relative merits of the triangle pixel splitting approach.
In essence, the code attempts to improve the image by enriching the local patterns that the CNN can learn from, potentially leading to better feature extraction and improved performance in computer vision tasks. However, empirical validation is needed to confirm these benefits
