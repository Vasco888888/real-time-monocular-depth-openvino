import urllib.request
import os

if not os.path.exists('models'):
    os.makedirs('models')

base_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
files = ["MiDaS_small.xml", "MiDaS_small.bin"]

for file in files:
    print(f"Downloading {file}...")
    urllib.request.urlretrieve(base_url + file, f"models/{file}")

print("\nInstalled MiDaS Small model.")