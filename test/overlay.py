import vlkit, cv2
from vlkit.image import normalize_uint8
print(vlkit.__file__)

from vlkit.visualization import overlay_heatmap
import matplotlib.pyplot as plt

# Load image and mask
image = cv2.imread("/media/ssd2t/Code/experiments/microus/087_42.image.jpg")
heatmap = cv2.imread("/media/ssd2t/Code/experiments/microus/087_42.pred.png", cv2.IMREAD_GRAYSCALE)
heatmap = heatmap / 255

# Overlay mask
overlay = overlay_heatmap(image, heatmap, alpha=0.3, threshold=0.5)

fig, axes = plt.subplots(1, 3)

for i, j in enumerate((image, heatmap, overlay)):
    axes[i].imshow(cv2.cvtColor(normalize_uint8(j), cv2.COLOR_BGR2RGB))

plt.show()

