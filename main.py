from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import torch
import torchvision.transforms as transforms
from skimage import color, segmentation, filters
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import pickle
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
category_model = torch.load("models/googlenet_category_model.pth", map_location=device)
category_model.eval()

with open("models/color_extraction.pkl", "rb") as file:
    look = pickle.load(file)

categories = ['Apparel', 'Accessories', 'Footwear', 'Personal Care', 'Free Items', 'Sporting Goods', 'Home']
colors = ['Black', 'White', 'Blue', 'Brown', 'Grey', 'Red', 'Green', 'Pink', 'Navy Blue', 'Purple', "Other"]

Others = ['Lavender', 'Grey Melange', 'Silver', 'Sea Green', 'Yellow', 'Rust', 'Magenta', 'Fluorescent Green', 'nan',
          'Turquoise Blue', 'Peach', 'Steel', 'Coffee Brown', 'Cream', 'Mustard', 'Nude', 'Off White', 'Beige', 'Teal',
          'Lime Green', 'Metallic', 'Bronze', 'Gold', 'Copper', 'Rose', 'Skin', 'Olive', 'Maroon', 'Orange',
          'Khaki', 'Charcoal', 'Tan', 'Taupe', 'Mauve', 'Burgundy', 'Mushroom Brown', 'Multi']


app = FastAPI()

# Define the image preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_batch

def segmentations(image):
    gray_image = color.rgb2gray(image)
    # Use a simple thresholding technique to get initial seeds
    seeds = gray_image < filters.threshold_otsu(gray_image)
    # Use region growing for segmentation
    labels = segmentation.flood(seeds, (0, 0), connectivity=2)

    # startRow, startColumn, endRow, endColumn
    startRow = -1
    startCol = -1
    endRow = -1
    endCol = -1

    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if labels[i][j] == False:
                if startRow == -1:
                    startRow = i
                endRow = i

    for i in range(len(labels[0])):
        for j in range(len(labels)):
            if labels[j][i] == False:
                if startCol == -1:
                    startCol = i
                endCol = i
    return (startCol, startRow, endCol, endRow)

# k-means
def extract_dominant_colors(image_path, k=1):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors


def extract_color(path):
    original_image = cv2.imread(path)
    (left, top, right, bottom) = segmentations(original_image)
    cropped_image = original_image[top:bottom+1, left:right+1]
    cv2.imwrite("cropped_img.jpg", cropped_image)
    
    ans = extract_dominant_colors("cropped_img.jpg")
    
    subs = []
    for i in range(len(colors)):
        sums = 0
        for x in range(1):
            for y in range(3):
                sums += abs(ans[x][y] - look[i][x][y])
        subs.append(sums)
    ind = subs.index(min(subs))
    if colors[ind] == "Other":
        return random.choice(Others)
    return colors[ind]

# Example: Upload image
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # Ensure that the file is an image (you can customize this check based on your requirements)
    allowed_image_extensions = {"png", "jpg", "jpeg", "gif"}
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() not in allowed_image_extensions:
        raise HTTPException(status_code=400, detail="Invalid file format. Only images are allowed.")

    # # Save the uploaded file (you might want to customize the path)
    with open(file.filename, "wb") as f:
        f.write(file.file.read())

    input_tensor = preprocess_image(file.filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output1 = category_model(input_tensor)
    
    probabilities_category = torch.nn.functional.softmax(output1[0], dim=0)
    color = extract_color(file.filename)

    os.remove(file.filename)
    os.remove("cropped_img.jpg")

    return {"Category": categories[torch.argmax(probabilities_category)], 
            "Color" : color}
