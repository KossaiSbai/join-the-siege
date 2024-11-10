import os
from io import BytesIO

import pytest
from src.app import app, allowed_file

# Define the folder where test files are stored
FILES_FOLDER = 'files'

# Expected classifications for each file in the `files` folder
EXPECTED_CLASSIFICATIONS = {
    "bank_statement_1.pdf": "bank_statement",
    "bank_statement_2.pdf": "bank_statement",
    "bank_statement_3.pdf": "bank_statement",
    "invoice_1.pdf": "invoice",
    "invoice_2.pdf": "invoice",
    "invoice_3.pdf": "invoice",
    "drivers_license_1.jpg": "drivers_license",
    "drivers_license_2.jpg": "drivers_license",
    "drivers_license_3.jpg": "drivers_license",
    # Add more files and expected classes as needed
}

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.mark.parametrize("filename, expected", [
    ("file.pdf", True),
    ("file.png", True),
    ("file.jpg", True),
    ("file.txt", False),
    ("file", False),
])
def test_allowed_file(filename, expected):
    assert allowed_file(filename) == expected

def test_no_file_in_request(client):
    response = client.post('/classify_file')
    assert response.status_code == 400

def test_no_selected_file(client):
    data = {'file': (BytesIO(b""), '')}  # Empty filename
    response = client.post('/classify_file', data=data, content_type='multipart/form-data')
    assert response.status_code == 400

@pytest.mark.parametrize("file_name, expected_class", EXPECTED_CLASSIFICATIONS.items())
def test_file_classification(client, file_name, expected_class):
    file_path = os.path.join(FILES_FOLDER, file_name)
    
    if not os.path.isfile(file_path):
        pytest.fail(f"Test file {file_name} does not exist in {FILES_FOLDER}")

    with open(file_path, 'rb') as file:
        data = {'file': (BytesIO(file.read()), file_name)}
        response = client.post('/classify_file', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    response_data = response.get_json()
    assert response_data["file_class"] == expected_class, f"Classification for {file_name} was incorrect"
