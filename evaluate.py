import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
def load_model(model_path='model/deepfake_model.pth'):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess image
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image.convert('RGB')).unsqueeze(0)

# Evaluate images from nested folders
def evaluate_from_nested_images(image_root='frames'):
    model = load_model()
    true_labels = []
    pred_labels = []
    label_map = {'real_video': 0, 'fake_video': 1}

    for label_name in ['real_video', 'fake_video']:
        main_folder = os.path.join(image_root, label_name)
        if not os.path.exists(main_folder):
            print(f"Missing folder: {main_folder}")
            continue

        # Loop through subfolders inside real_video or fake_video
        for subfolder in os.listdir(main_folder):
            subfolder_path = os.path.join(main_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for file in os.listdir(subfolder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(subfolder_path, file)
                    try:
                        image = Image.open(path)
                        input_tensor = transform_image(image)
                        with torch.no_grad():
                            output = model(input_tensor)
                            _, predicted = torch.max(output, 1)
                            pred_labels.append(predicted.item())
                            true_labels.append(label_map[label_name])
                    except Exception as e:
                        print(f"Failed to process image {path}: {e}")

    if not true_labels:
        print("No images found for evaluation.")
        return

    # Report and Confusion Matrix
    print("\nClassification Report:\n", classification_report(true_labels, pred_labels, target_names=['Real', 'Fake']))
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Image-level)')
    plt.show()

if __name__ == '__main__':
    evaluate_from_nested_images()
