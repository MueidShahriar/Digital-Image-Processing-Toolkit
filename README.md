# Digital Image Processing Toolkit

A simple desktop application for performing basic digital image processing operations. This toolkit allows users to upload an image, apply various transformations or enhancements, and save the processed output. It is designed for students, researchers, and anyone learning about image processing concepts.

---

## ✨ Features

* **Upload & Save Image**
  Load an image into the application and export the processed result.

* **Reset Image**
  Restore the original image at any time.

* **Image Operations**

  * Negative Transformation
  * Sharpening
  * Edge Detection
  * Histogram Display

* **Filtering & Enhancements**

  * **Smoothing (Kernel-based filter)**
  * **Log Transformation (C factor adjustable)**
  * **Thresholding (Binary conversion with adjustable threshold)**
  * **Gamma Transformation (Gamma correction with adjustable gamma value)**

* **Resize Image**
  Resize the image to any specified width and height.

---

## 🖥️ GUI Layout

* **Left Panel** → Developer info
* **Center** → Displays the *Original Image*
* **Right** → Displays the *Processed Image*
* **Bottom Panel** → Controls for applying filters, transformations, and resizing

---

## 🚀 Getting Started

### Prerequisites

* Python 3.x
* Required libraries (install using pip):

  ```bash
  pip install numpy opencv-python pillow matplotlib
  ```

### Running the Application

```bash
python main.py
```

---

## 📸 Example Usage

1. Upload an image using **Upload Image**.
2. Select an operation (e.g., Apply Smooth, Apply Log, Apply Threshold, etc.).
3. View the processed result in the **Processed Image** window.
4. Save the processed image if needed.

---

## 📂 Repository Structure

```
├── main.py                # Main application file
├── assets/                # (Optional) Icons, test images
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## 👨‍💻 Developer

**MD. MUEID SHAHRIAR**
Student, BAUET

---

## 📜 License

This project is licensed under the MIT License – feel free to use and modify.

---

## 🙌 Contribution

Pull requests are welcome! If you’d like to contribute, please fork the repository and submit a PR.
