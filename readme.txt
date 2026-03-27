
 🧠 Face Recognition Preprocessing Pipeline (MTCNN + OpenCV)



 📌 Overview

This project focuses on building a clean preprocessing pipeline for face recognition. It uses MTCNN (MultiTask Cascaded Convolutional Networks) to detect faces in images, extracts them, and prepares them for training deep learning models like FaceNet.

The main goal is to take a raw image dataset, automatically detect faces, crop them, resize them to a fixed input size, and organize them into labeled NumPy arrays ready for model training.



 🚀 What This Project Does

This pipeline handles the full preprocessing stage for face recognition:

 Loads images from a structured dataset folder
 Detects faces using MTCNN
 Extracts and crops detected faces
 Resizes faces to 160×160 (FaceNetcompatible input size)
 Labels each face based on folder name
 Converts everything into NumPy arrays for training
 Provides visualization of processed faces



 🛠 Tech Stack

 Python
 OpenCV
 TensorFlow
 MTCNN (Face Detection)
 NumPy
 Matplotlib
 Google Colab



 📂 Dataset Structure

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



 ⚙️ How It Works

 1. Load Dataset

The notebook mounts Google Drive and loads images directly from the dataset directory.



 2. Face Detection (MTCNN)

Each image is passed through MTCNN, which returns bounding box coordinates:

 x, y, width, height



 3. Face Extraction

The detected region is cropped from the image and isolated as the face.



 4. Preprocessing

Each face is:

 Converted from BGR → RGB
 Resized to 160×160 pixels
 Stored as a NumPy array

This ensures compatibility with FaceNetstyle models.



 🧩 Core Component: `FACELOADING` Class

The `FACELOADING` class automates the entire pipeline.



 🔹 `extract_face(filename)`

 Reads an image
 Detects face using MTCNN
 Crops and resizes the face



 🔹 `load_faces(dir)`

 Loads all images in a folder
 Extracts faces
 Handles errors gracefully



 🔹 `load_classes()`

 Iterates through all folders (labels)
 Builds:

   `X` → face images
   `Y` → labels (person names)
 Returns NumPy arrays for training



 🔹 `plot_images()`

 Displays processed faces in a grid
 Used for visualization and verification



 📦 Output Format

After processing:

 `X` → Face images → `(160, 160, 3)`
 `Y` → Labels (person names)

This is ready for:

 Face recognition training
 FaceNet embeddings
 Classification models



 💻 Example Workflow

```python
faceloading = FACELOADING(dataset_path)

X, Y = faceloading.load_classes()

faceloading.plot_images()
```



 🖼 Visualization

After preprocessing, the system displays detected and cropped faces in a grid format to verify detection quality.

> 📌 Insert your screenshots here:

```md
![Result 1](pic1.png)
![Result 2](pic2.png)
```



 ⚠️ Notes

 Assumes at least one face per image
 If multiple faces exist, only the first detected face is used
 Images without detectable faces are skipped
 Works best with clear, frontfacing images



 🔧 Possible Improvements

 Handle multiple faces per image
 Add FaceNet embedding generation
 Improve dataset balancing
 Add data augmentation (rotation, brightness, etc.)
 Save processed dataset for reuse
 Build realtime webcam face recognition



 📊 Summary

This project provides a solid preprocessing foundation for face recognition systems. It converts raw image data into a structured, machinelearningready dataset using reliable face detection and clean automation.
