import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import zipfile

# Base folder
base_dir = "cats_dogs_dataset"
os.makedirs(base_dir, exist_ok=True)

categories = ["cats", "dogs"]
num_images = 10

# Create category folders
for cat in categories:
    os.makedirs(os.path.join(base_dir, cat), exist_ok=True)

# Function to generate dummy images
def create_image(filepath, label, color):
    img = Image.new("RGB", (128, 128), color=color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((40, 55), label, fill=(255, 255, 255), font=font)
    img.save(filepath, "JPEG")

rows = []
colors = {
    "cats": (150, 50, 50),  # reddish
    "dogs": (50, 50, 150)   # bluish
}

# Generate images + labels
for cat in categories:
    for i in range(1, num_images + 1):
        filename = f"{cat}{i}.jpg"
        filepath = os.path.join(base_dir, cat, filename)
        create_image(filepath, cat[:-1].upper(), colors[cat])
        rows.append([f"{cat}/{filename}", cat[:-1]])

# Save labels.csv
labels_df = pd.DataFrame(rows, columns=["filename", "target"])
labels_df.to_csv(os.path.join(base_dir, "labels.csv"), index=False)

# --- ZIP the dataset ---
zip_path = "cats_dogs_dataset.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, base_dir)
            zipf.write(filepath, arcname)

print("Cats vs Dogs dataset created and zipped successfully!")
print(f"📂 Dataset folder: {base_dir}")
print(f"📦 Zipped file: {zip_path}")
