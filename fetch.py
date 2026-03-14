import random
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def get_random_artwork():
    # Search Met Museum for public domain paintings with images
    search = requests.get(
        "https://collectionapi.metmuseum.org/public/collection/v1/search",
        params={"hasImages": True, "isPublicDomain": True, "q": "painting"},
        timeout=10
    )
    search.raise_for_status()
    object_ids = search.json().get("objectIDs", [])
    random.shuffle(object_ids)
    for oid in object_ids[:20]:
        obj = requests.get(
            f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}",
            timeout=10
        )
        obj.raise_for_status()
        data = obj.json()
        if data.get("primaryImageSmall"):
            return data
    return None

try:
    artwork = get_random_artwork()
    if not artwork:
        print("No artwork found.")
    else:
        response = requests.get(artwork["primaryImageSmall"], timeout=15)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")

        print(f"Title:  {artwork.get('title', 'Unknown')}")
        print(f"Artist: {artwork.get('artistDisplayName', 'Unknown')}")
        print(f"Date:   {artwork.get('objectDate', 'Unknown')}")

        plt.imshow(image)
        plt.axis("off")
        plt.title(artwork.get("title", ""))
        plt.show()

        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        print("Description:", processor.decode(out[0], skip_special_tokens=True))

except requests.exceptions.RequestException as e:
    print("Request failed:", e)
