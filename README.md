# Face Mask Detection System

## Overview
The Face Mask Detection System is an AI-driven solution designed to identify whether individuals are wearing masks in images or real-time video streams. It leverages Convolutional Neural Networks (CNNs) for classification and MTCNN for accurate face detection. This project demonstrates the application of deep learning in promoting public health and safety, particularly during pandemics.

The primary objective of this project is to build a system that classifies images into two categories: "face with mask" and "face without mask." The system is powered by a CNN model trained on a dataset containing images of faces, both with and without masks. The process involves loading and preprocessing the images, training the CNN, and using the trained model to make predictions.

## Features
- Real-time detection of face masks using webcam or images.
- Bounding box visualization around detected faces.
- Highly accurate model trained on annotated datasets.
- Scalable and adaptable for integration into surveillance systems.

## Dataset and Image Processing

### **A. Loading and Exploring the Dataset:**

The dataset contains images and annotations for each image that describe the bounding box of the face. The images are stored in a folder, while annotations are provided in a CSV file. The following code snippet provides insight into the dataset:

- Images: Stored as .jpg files in a directory.

- Annotations: A CSV file containing details of the bounding boxes for faces in the images, such as x1, x2, y1, y2, and the class (face_with_mask or face_no_mask).
  
The dataset is divided into training and testing sets. The training set includes images starting from index 1698 onward, while the test set consists of the first 1698 images.

### **B. Bounding Box Generation**
The bounding box for each image is defined by the coordinates (x1, y1) for the top-left corner and (x2, y2) for the bottom-right corner. The function get_boxes is used to extract the bounding boxes for each image based on the annotations.

Bounding boxes are drawn around the faces in the images to verify the locations where faces are detected. This helps ensure that the correct part of the image is used for training the model.

### **C. Data Preprocessing**

Before training, the images are preprocessed to fit the input requirements of the CNN:

- Grayscale Conversion: The images are loaded in grayscale mode.
  
- Cropping and Resizing: The bounding box coordinates are used to crop the face from the image. The cropped face is resized to a standard size (50x50 pixels) to ensure uniformity in input size for the model.

- Data Normalization: The pixel values of the images are normalized to range between 0 and 1.

## Model Architecture
The core of the system is a Convolutional Neural Network (CNN), a class of deep learning models well-suited for image classification tasks. The CNN architecture used here consists of the following layers:

### **Convolutional Layers:** 
These layers apply convolutional operations to detect features like edges and textures in the images.

- The first convolutional layer uses 100 filters with a 3x3 kernel and ReLU activation.
- The second convolutional layer uses 64 filters with a 3x3 kernel and ReLU activation.

### **Max-Pooling Layers:** 
After each convolutional layer, a max-pooling operation is performed to reduce the spatial dimensions of the image, effectively downsampling it.

### **Flattening:**
After the convolutional and pooling layers, the 2D features are flattened into a 1D vector, which can be passed to fully connected layers.

### **Fully Connected Layers:**
These layers help in making final predictions:

- A dense layer with 50 units and ReLU activation is used to process the flattened features.
- A dropout layer with a rate of 0.2 is applied to prevent overfitting.
- The final output layer has 2 units (for "face_with_mask" and "face_no_mask") and uses softmax activation for classification.

### **Compilation:**
The model is compiled using the Adam optimizer with a learning rate of 1e-3 and decay of 1e-5. The loss function used is categorical_crossentropy since it is a multi-class classification problem.

### **Training:** 
The model is trained on the preprocessed image data for 50 epochs with a batch size of 10 and is then saved using model.save('model_02.h5') for future use in making predictions.

## Face Detection and Mask Prediction

### **Face Detection:**
Face detection is performed using the MTCNN (Multi-task Cascaded Convolutional Networks) algorithm. MTCNN is capable of detecting faces along with facial landmarks such as eyes, nose, and mouth, which are important for this task. The detect_faces_in_image function uses MTCNN to locate faces in the image.

### **Mask Prediction on Faces:**
Once the face is detected, the bounding box of the face is used to crop the image and pass it to the trained CNN model for mask prediction. The cropped face is resized to 50x50 pixels, converted to grayscale, and reshaped to match the input shape of the CNN.

- **Prediction Output:** The model outputs a class (either 0 for "no mask" or 1 for "mask"), which is used to display the result on the image.

### **Displaying the Results:**
The image with the predicted mask status is displayed using matplotlib. If no face is detected, a message is printed.


## **Testing the Model on a Single Image:**
The final step involves testing the model on a specific image to verify its performance. The function test_single_image allows testing a single image by specifying its path and calling the display function.


## Limitations and Future Enhancements
### Limitations
- Reduced accuracy in low-light or occluded scenarios.
- Computationally intensive for real-time performance.

### Future Enhancements
- Expand dataset to include diverse images (e.g., different age groups, ethnicities).
- Optimize for deployment on edge devices like Raspberry Pi.
- Add multi-face detection capabilities.

### Contributors
**Abashesh Ranabhat**

Feel free to contribute to this project by forking the repository and creating pull requests.

### License

This project is licensed under the MIT License.

### Acknowledgments

Special thanks to open-source libraries like TensorFlow, OpenCV, and MTCNN for enabling this project.
