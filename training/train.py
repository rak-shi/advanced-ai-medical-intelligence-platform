import os
import copy
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
VALIDATION_SIZE = 0.20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "chest_xray"
)

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)

TEST_DIR = os.path.join(
    DATA_DIR,
    "test"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "trained_model"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "training",
    "outputs"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "pneumonia_efficientnet_b0.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("ADVANCED AI MEDICAL INTELLIGENCE PLATFORM")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Dataset: {DATA_DIR}")


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


evaluation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading dataset...")


# Training copy with augmentation
full_train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)


# Validation copy without augmentation
full_validation_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=evaluation_transform
)


# Independent test dataset
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=evaluation_transform
)


class_names = full_train_dataset.classes

print(f"\nClasses: {class_names}")

print(
    f"Class mapping: "
    f"{full_train_dataset.class_to_idx}"
)


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

targets = np.array(
    full_train_dataset.targets
)

indices = np.arange(
    len(full_train_dataset)
)


train_indices, validation_indices = train_test_split(

    indices,

    test_size=VALIDATION_SIZE,

    stratify=targets,

    random_state=SEED
)


train_dataset = Subset(
    full_train_dataset,
    train_indices
)

validation_dataset = Subset(
    full_validation_dataset,
    validation_indices
)


print("\nDataset split:")

print(
    f"Training:   {len(train_dataset)}"
)

print(
    f"Validation: {len(validation_dataset)}"
)

print(
    f"Testing:    {len(test_dataset)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_labels = targets[train_indices]

validation_labels = targets[validation_indices]


print("\nTraining class distribution:")

for index, class_name in enumerate(class_names):

    count = np.sum(
        train_labels == index
    )

    print(
        f"{class_name}: {count}"
    )


print("\nValidation class distribution:")

for index, class_name in enumerate(class_names):

    count = np.sum(
        validation_labels == index
    )

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print(
    "\nLoading pretrained EfficientNet-B0..."
)


weights = models.EfficientNet_B0_Weights.DEFAULT


model = models.efficientnet_b0(
    weights=weights
)


# ============================================================
# FREEZE FEATURE EXTRACTION LAYERS
# ============================================================

for parameter in model.features.parameters():

    parameter.requires_grad = False


# ============================================================
# REPLACE CLASSIFICATION LAYER
# ============================================================

number_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    number_features,
    len(class_names)
)


model = model.to(DEVICE)


print(
    "EfficientNet-B0 loaded successfully."
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(

    model.classifier.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
validation_losses = []

train_accuracies = []
validation_accuracies = []

best_validation_accuracy = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)

print("STARTING MODEL TRAINING")

print("=" * 60)


training_start_time = time.time()


for epoch in range(NUM_EPOCHS):

    epoch_start_time = time.time()

    print(
        f"\nEpoch "
        f"{epoch + 1}/{NUM_EPOCHS}"
    )

    print("-" * 50)


    # ========================================================
    # TRAIN PHASE
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for batch_number, (
        images,
        labels
    ) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predictions = torch.max(
            outputs,
            1
        )


        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


        if (
            batch_number + 1
        ) % 50 == 0:

            print(
                f"Processed batch "
                f"{batch_number + 1}/"
                f"{len(train_loader)}"
            )


    train_loss = (
        running_loss
        / len(train_dataset)
    )

    train_accuracy = (
        correct / total
    )


    # ========================================================
    # VALIDATION PHASE
    # ========================================================

    model.eval()

    validation_running_loss = 0.0

    validation_correct = 0
    validation_total = 0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


            validation_running_loss += (
                loss.item()
                * images.size(0)
            )


            _, predictions = torch.max(
                outputs,
                1
            )


            validation_correct += (
                predictions == labels
            ).sum().item()

            validation_total += (
                labels.size(0)
            )


    validation_loss = (
        validation_running_loss
        / len(validation_dataset)
    )


    validation_accuracy = (
        validation_correct
        / validation_total
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    train_losses.append(
        train_loss
    )

    validation_losses.append(
        validation_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    validation_accuracies.append(
        validation_accuracy
    )


    epoch_time = (
        time.time()
        - epoch_start_time
    )


    print(
        f"\nTrain Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time / 60:.2f} minutes"
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if (
        validation_accuracy
        > best_validation_accuracy
    ):

        best_validation_accuracy = (
            validation_accuracy
        )

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        print(
            "✓ New best model found."
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)


# ============================================================
# SAVE CHECKPOINT
# ============================================================

checkpoint = {

    "architecture":
        "efficientnet_b0",

    "model_state_dict":
        model.state_dict(),

    "class_names":
        class_names,

    "class_to_idx":
        full_train_dataset.class_to_idx,

    "image_size":
        IMAGE_SIZE,

    "best_validation_accuracy":
        best_validation_accuracy
}


torch.save(
    checkpoint,
    MODEL_PATH
)


print("\n" + "=" * 60)

print("MODEL SAVED")

print("=" * 60)

print(MODEL_PATH)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating independent test dataset...")


model.eval()


true_labels = []
predicted_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )


        true_labels.extend(
            labels.numpy()
        )

        predicted_labels.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


precision = precision_score(
    true_labels,
    predicted_labels,
    average="binary",
    pos_label=1,
    zero_division=0
)


recall = recall_score(
    true_labels,
    predicted_labels,
    average="binary",
    pos_label=1,
    zero_division=0
)


f1 = f1_score(
    true_labels,
    predicted_labels,
    average="binary",
    pos_label=1,
    zero_division=0
)


print("\n" + "=" * 60)

print("FINAL TEST RESULTS")

print("=" * 60)


print(
    f"Accuracy : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score : "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:\n"
)


report = classification_report(

    true_labels,

    predicted_labels,

    target_names=class_names,

    zero_division=0
)


print(report)


# ============================================================
# SAVE RESULTS AS TEXT
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_results.txt"
)


with open(
    results_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Advanced AI Medical Intelligence Platform\n"
    )

    file.write(
        "Pneumonia Detection - EfficientNet-B0\n\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: "
        f"{precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall: "
        f"{recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score: "
        f"{f1 * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


print("\nConfusion Matrix:")

print(cm)


plt.figure(
    figsize=(6, 5)
)

plt.imshow(cm)

plt.title(
    "Pneumonia Detection Confusion Matrix"
)

plt.colorbar()


ticks = np.arange(
    len(class_names)
)


plt.xticks(
    ticks,
    class_names
)

plt.yticks(
    ticks,
    class_names
)


plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


for row in range(
    cm.shape[0]
):

    for column in range(
        cm.shape[1]
    ):

        plt.text(

            column,

            row,

            str(
                cm[row, column]
            ),

            horizontalalignment="center",

            verticalalignment="center"
        )


plt.tight_layout()


confusion_matrix_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    confusion_matrix_path,
    dpi=300
)

plt.close()


# ============================================================
# ACCURACY CURVE
# ============================================================

epochs = range(
    1,
    NUM_EPOCHS + 1
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)


plt.plot(
    epochs,
    validation_accuracies,
    marker="o",
    label="Validation Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.grid()


plt.tight_layout()


accuracy_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_curve.png"
)


plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training Loss"
)


plt.plot(
    epochs,
    validation_losses,
    marker="o",
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid()

plt.tight_layout()


loss_path = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)


plt.savefig(
    loss_path,
    dpi=300
)

plt.close()


# ============================================================
# FINISH
# ============================================================

total_training_time = (
    time.time()
    - training_start_time
)


print("\n" + "=" * 60)

print("TRAINING COMPLETED")

print("=" * 60)


print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy * 100:.2f}%"
)


print(
    f"Total Training Time: "
    f"{total_training_time / 60:.2f} minutes"
)


print("\nGenerated files:")

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Results: {results_path}"
)

print(
    f"Confusion Matrix: "
    f"{confusion_matrix_path}"
)

print(
    f"Accuracy Curve: "
    f"{accuracy_path}"
)

print(
    f"Loss Curve: "
    f"{loss_path}"
)