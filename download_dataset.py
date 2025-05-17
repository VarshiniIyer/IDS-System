import requests

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"{filename} downloaded successfully!")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

# URLs of raw dataset files on GitHub
train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

# Download files
download_file(train_url, "KDDTrain+.txt")
download_file(test_url, "KDDTest+.txt")
