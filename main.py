from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
category_model = torch.load("models/googlenet_category_model.pth", map_location=device)
color_model = torch.load("models/googlenet_color_model.pth", map_location=device)
category_model.eval()
color_model.eval()

categories = ['Apparel', 'Accessories', 'Footwear', 'Personal Care', 'Free Items', 'Sporting Goods', 'Home']
colors = ['Black', 'White', 'Blue', 'Brown', 'Grey', 'Red', 'Green', 'Pink', 'Navy Blue', 'Purple']

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
        output2 = color_model(input_tensor)
    
    probabilities_category = torch.nn.functional.softmax(output1[0], dim=0)
    probabilities_color = torch.nn.functional.softmax(output2[0], dim=0)

    category = categories[torch.argmax(probabilities_category)]
    color_ind = torch.argmax(probabilities_color)
    color = ""
    if color_ind == len(colors):
        color = random.choice(Others)
    else:
        color = colors[color_ind]

    os.remove(file.filename)

    return {"Category": category, 
            "Color" : color}
