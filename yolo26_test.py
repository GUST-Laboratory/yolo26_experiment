from pathlib import Path
from ultralytics import YOLO

# Get the base directory of the current file
BASE_DIR = Path(__file__).resolve().parent

# Specify the path to the YOLOv8 model weights
WEIGHTS = BASE_DIR / "yolo26n.pt"

# Check if the weights file exists
if not WEIGHTS.is_file():
    raise FileNotFoundError(f"找不到本地权重文件：{WEIGHTS}")

# Load the YOLOv8 model with the specified weights
model = YOLO(str(WEIGHTS))

# Perform inference on an image
results = model.predict(
    source=str("demo.jpg"),
    save=True,
    save_txt=True,
    save_conf=True,
    nms=False,
    conf=0.25,
    iou=0.45,
    device="mps",
    project=str(BASE_DIR / "output"),
    name="test"
)

# Print the results
print(results)
# results[0].show()  # Display the image with detected objects
