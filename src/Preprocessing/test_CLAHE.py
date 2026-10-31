
# Import the CLAHE function from our preprocessing file
from ImagePreprocessing import apply_clahe

# Import Pillow to open an image
from PIL import Image

# Import Matplotlib to display images
import matplotlib.pyplot as plt

# Load one handwritten image
image = Image.open("sample.png").convert("L")

# Apply CLAHE preprocessing
enhanced_image = apply_clahe(image)

# Display original and enhanced images for comparison
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap="gray")
plt.title("CLAHE Enhanced Image")
plt.axis("off")

plt.tight_layout()
plt.show()
