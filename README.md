📌 ImageOriginClassifier
🖼️ What This Project Does
ImageOriginClassifier is a neural network classifier that determines the origin of images. It is trained on three categories:
✅ Stalker (in-game screenshots from S.T.A.L.K.E.R.)
✅ Minecraft (screenshots from Minecraft)
✅ Real Photos (pictures from the real world)

🎯 Key Features:

Image classification using PyTorch
Confidence-based predictions
The ability to reject images that do not belong to any of the known classes
Automated dataset collection and preprocessing
🛠️ Tech Stack:

Python 3.12
PyTorch
Torchvision
PIL
NumPy
Matplotlib
🔹 Dataset Preparation Script
This script automatically organizes images into dataset/train/ and dataset/test/ folders by copying them from a raw dataset directory.
💡 How to Use:

Place your raw images into raw_dataset/Stalker/, raw_dataset/Minecraft/, and raw_dataset/Real/.
Run the script to automatically split and organize the dataset.
Train your model using the prepared dataset.
🚀 Future Improvements:

Expand dataset with more diverse images
Improve model accuracy with data augmentation
Optimize classification confidence threshold
Let me know if you need any modifications! 😊
