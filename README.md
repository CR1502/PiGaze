# PiGaze
Raspberry Pi-based eye movement tracking using Pi Camera, CNN, and PyTorch
## Eye Movement Tracking with Raspberry Pi

This project uses a Raspberry Pi, Pi Camera, and Convolutional Neural Networks (CNN) with PyTorch to track eye movements in real-time.

### Features
- Real-time face and eye detection
- CNN-based eye movement classification

### Requirements
- Raspberry Pi 3/4
- Pi Camera
- PyTorch, OpenCV, Picamera, etc.

### Dataset

This project uses [MPIIFaceGaze Dataset]https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3240 and shape_predictor_68_face_landmarks.dat file https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat


- Download the dataset through the link above
- Click "Acess Dataset" and then "Download ZIP"
- Accept the dataset terms and download
- Extract the contents of the dataset into the `data/dataset/` directory  

### BibTeX Citation
If you use this dataset or project, please cite the following:

```bibtex
@data{darus-3240_2023,
author = {Bulling, Andreas},
publisher = {DaRUS},
title = {{MPIIFaceGaze}},
year = {2023},
version = {V1},
doi = {10.18419/darus-3240},
url = {https://doi.org/10.18419/darus-3240}
}

### Installation
```bash
git clone https://github.com/CR1502/PiGaze.git
cd eye-movement-tracking
pip install -r requirements.txt
```
### Acknowledgments
- MPIIFaceGaze Dataset: For providing robust training data.
- Dlib: For facial landmark detection.
