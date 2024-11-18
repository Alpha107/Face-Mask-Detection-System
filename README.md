# Face Mask Detection System

## Overview
The Face Mask Detection System is a deep learning-based solution to detect whether individuals are wearing masks in images or real-time video feeds. It uses convolutional neural networks (CNNs) for classification and MTCNN for face detection. This project is a practical implementation of AI in ensuring public health and safety, especially in the context of pandemics.

## Features
- Real-time detection of face masks using webcam or images.
- Bounding box visualization around detected faces.
- Highly accurate model trained on annotated datasets.
- Scalable and adaptable for integration into surveillance systems.

## Project Structure

- **Dataset:**
Annotated images with bounding boxes for mask presence.
Includes categories: face_with_mask and face_no_mask.

- **Model:**
Custom CNN architecture for mask classification.

- **Face Detection:**
Utilizes MTCNN for region-of-interest extraction.

- **Augmentation:**
Applies transformations like rotation, flipping, and shifting to enhance model robustness.

## Results
The model achieved the following results on the test dataset:

- Accuracy: 97%
- Confusion Matrix:
- True Positives: Correctly detected masks.
- False Positives: Incorrectly detected masks.
- Sample Detection Output

## Performance Metrics
- Accuracy: Measures overall correctness.
- Precision: Evaluates the correctness of positive predictions.
- Recall: Measures how well the model detects all positive cases.
- F1-Score: Balances precision and recall.

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
