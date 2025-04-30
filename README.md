# Handedness Detection Using Machine Learning  
**ECEN 360 Final Project – Texas A&M University**

This project implements a real-time machine learning pipeline to detect whether a hand is a **left or right hand** using webcam input and landmark data extracted via MediaPipe. Unlike MediaPipe's built-in handedness classifier, this model relies solely on the geometric structure of the hand — making it more robust to hand orientation and camera position.

The system collects 21 landmark points per hand (63 features), trains a **Random Forest Classifier**, and integrates the trained model into a live OpenCV pipeline for real-time classification.

## 🧠 Project Features
- Real-time hand tracking using **MediaPipe**
- Landmark-based feature extraction (21 3D points → 63 features)
- Custom-trained **Random Forest Classifier** using scikit-learn
- Left vs. right hand prediction independent of screen position
- Live video inference using **OpenCV** with model integration
- Dataset generation pipeline and model persistence with **Joblib**

## 🛠 Technologies Used
- Python  
- MediaPipe  
- OpenCV  
- Scikit-learn (RandomForestClassifier)  
- NumPy  
- Joblib

## 📁 Project Structure
- `collect_hand_data.py` – Tool to record labeled training data  
- `train_model.ipynb` – Jupyter notebook for training the classifier  
- `hand_classifier.pkl` – Saved model for inference  
- `main.py` – Real-time webcam classification pipeline  
- `README.md` – Project documentation  

## 🔍 Future Improvements
- Expand dataset diversity for improved accuracy  
- Add model confidence scoring and error handling  
- Explore deep learning alternatives (e.g., CNNs with raw image input)


