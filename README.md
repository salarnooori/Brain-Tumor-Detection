# Brain Tumor Detection

This notebook aims to detect brain tumors using image processing techniques and a simple neural network. The goal is to extract a 20-dimensional feature vector for each pixel and predict whether the pixel belongs to a tumor or the background.

## Dataset

The notebook uses the DRIVE dataset to prepare the data for model training.

## Steps to Prepare the Data for Model Training

1. **Setup Paths**:
   - Defines paths for images, masks, and labels from the DRIVE dataset.

2. **Load File Lists**:
   - Retrieves and sorts the filenames for images, masks, and labels from the specified directories.

3. **Initialize DataFrame**:
   - Creates an empty DataFrame to store the extracted features from each processed image.

4. **Image Processing Loop**:
   - Iterates over the first 20 images:
     - Reads each image and its corresponding mask and label.
     - Reshapes the mask and label arrays to standard dimensions.
     - Converts label values from 255 to 1 for binary representation.
     - Applies preprocessing to the image using the `PreProcess` function.

5. **Feature Extraction**:
   - Extracts various features from the preprocessed image:
     - Edge features using the `Edge_Algorithm_features` function.
     - Morphological features using the `Morphological_features` function.
     - Statistical features (mean, min, max, skewness, kurtosis, standard deviation, mean absolute deviation, and root sum of squares) using the `Statistical_features` function.
     - Gradient-based features (Gx, Gy, magnitude, and angle) using the `Gradient_Based_features` function.
     - Hessian features using the `Hessian_features` function.

6. **Feature Compilation**:
   - Stacks the extracted features along with the corresponding label and creates a DataFrame from the features.

7. **Data Storage**:
   - Concatenates the newly created DataFrame with the main DataFrame to accumulate all extracted features.

