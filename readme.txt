Face Recognition Preprocessing Pipeline (MTCNN + OpenCV)

 Overview

This project focuses on building a clean preprocessing pipeline for face recognition. It uses MTCNN (Multi-Task Cascaded Convolutional Networks) to detect faces in images, extracts them, and prepares them for training deep learning models like FaceNet.

The main goal is to take a raw image dataset, detect faces automatically, crop them, resize them to a fixed input size, and organize them into labeled arrays ready for model training.

---

 What This Project Does

This pipeline handles the full preprocessing stage for face recognition:

 Loads images from a structured dataset folder
 Detects faces using MTCNN
 Extracts and crops detected faces
 Resizes faces to 160×160 (FaceNet-compatible input size)
 Labels each face based on its folder name
 Converts everything into NumPy arrays for training
 Provides simple visualization of processed faces

---

 Tech Stack

 Python
 OpenCV
 TensorFlow (used for environment / future model integration)
 MTCNN (face detection)
 Matplotlib
 NumPy
 Google Colab (development environment)

---

 Dataset Structure

Your dataset should be organized like this:

```
dataset/
│
├── person_1/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── person_2/
│   ├── img1.jpg
│   ├── img2.jpg
│
└── person_3/
    ├── img1.jpg
```

Each folder name becomes the label for the images inside it.

---

 How It Works

 1. Load Dataset from Google Drive

The notebook mounts Google Drive and loads images directly from a dataset directory.

 2. Face Detection (MTCNN)

Each image is passed through MTCNN, which returns:

 Bounding box coordinates (x, y, width, height)

 3. Face Extraction

The detected region is cropped from the image and isolated as the face.

 4. Preprocessing

Each face is:

 Converted from BGR → RGB
 Resized to 160×160 pixels
 Stored as a NumPy array

This ensures compatibility with FaceNet-style models.

---

 Core Component: FACELOADING Class

The `FACELOADING` class automates the entire preprocessing pipeline.

 Key Functions

 `extract_face(filename)`

 Reads an image
 Detects face using MTCNN
 Crops and resizes the face

 `load_faces(dir)`

 Loads all images in a folder
 Extracts faces from each image
 Handles errors gracefully

 `load_classes()`

 Iterates through all folders (labels)
 Builds:

   `X` → face images
   `Y` → labels (person names)
 Returns NumPy arrays for training

 `plot_images()`

 Displays processed faces in a grid for visualization

---

 Output Format

After processing, the dataset becomes:

 `X`: Array of face images (160×160×3)
 `Y`: Corresponding labels (person names)

This is ready for:

 Face recognition training
 Embedding generation (FaceNet)
 Classification models

---

 Example Workflow

```python
faceloading = FACELOADING(dataset_path)
X, Y = faceloading.load_classes()
faceloading.plot_images()
```

---

 Visualization

After preprocessing, the system displays detected and cropped faces in a grid format so you can verify detection quality visually.

(You can insert your screenshots here)

---

 Notes

 The system assumes at least one face per image.
 If multiple faces exist, only the first detected face is used.
 Images without detectable faces are skipped.
 Works best with clear, front-facing images.

---

 Possible Improvements

If you want to extend this project, here are good next steps:

 Handle multiple faces per image
 Add face embedding generation (FaceNet model integration)
 Improve dataset balancing
 Add data augmentation (rotation, brightness, etc.)
 Save processed dataset to disk for reuse
 Build a real-time webcam face recognition system

---

 Summary

This project is a solid preprocessing foundation for any face recognition system. It takes raw image data and converts it into a structured, machine-learning-ready dataset using reliable face detection and clean automation.
