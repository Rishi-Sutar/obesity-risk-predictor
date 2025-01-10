import requests
import json

# Replace with your actual scoring URI
scoring_uri = 'http://a1711eff-863e-4480-81c0-635b9df18f5a.westus.azurecontainer.io/score'

# Example data to be scored
data = {
    "data": [
        [1, 3.48, 2.4503, 0.693147, 0.693147, 0.438397, 1.0890, 0.0, -0.081469, 0.0, 1.176486, 0.523111, -0.557096,  0.021108, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    ]
}

# Convert the data to JSON format
input_data = json.dumps(data)

# Set the content type
headers = {'Content-Type': 'application/json'}

# Make the request and display the response
response = requests.post(scoring_uri, data=input_data, headers=headers)
print(response.json())

