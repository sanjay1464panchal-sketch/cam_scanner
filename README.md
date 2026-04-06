# cam_scanner
# 📄 Document Scanner (CamScanner)

## 🚀 Overview

This project is a Python-based document scanner that converts images into clean, scanned documents similar to CamScanner.

## 🧠 Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib

## 🔍 Features

* Automatic edge detection
* Document contour detection
* Perspective transformation
* Clean black & white scan output

## 📸 Input

![Input](sample_images/input.jpg)

## 📄 Output

![Output](outputs/scanned_output.jpg)

## ⚙️ How It Works

1. Convert image to grayscale
2. Apply edge detection (Canny)
3. Detect largest rectangular contour
4. Apply perspective transform
5. Convert to scanned (thresholded) output

## ▶️ How to Run

```bash
pip install -r requirements.txt
python scanner.py
```

## 💡 Future Improvements

* OCR (Text extraction)
* Real-time scanning via webcam
* Deep learning-based detection

## 👩‍💻 Author Priyanshi
