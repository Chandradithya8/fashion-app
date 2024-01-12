import io
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture
def image_file():
    # Replace with the path to a sample image file for testing
    sample_image_path = "images/image.jpg"
    with open(sample_image_path, "rb") as f:
        content = f.read()
        return {"file": ("image.jpg", io.BytesIO(content))}

def test_create_upload_file_valid_image(image_file):
    response = client.post("/uploadfile/", files=image_file)
    assert response.status_code == 200
    assert "Category" in response.json()
    assert "Color" in response.json()

def test_create_upload_file_invalid_file_format():
    invalid_file = {"file": ("invalid.txt", "some content")}
    response = client.post("/uploadfile/", files=invalid_file)
    assert response.status_code == 400
    assert "Invalid file format. Only images are allowed." in response.text
