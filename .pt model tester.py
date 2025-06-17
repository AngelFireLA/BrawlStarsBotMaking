from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from ultralytics.engine.results import Results
from tkinter import Tk
from tkinter.filedialog import askopenfilename


Tk().withdraw()


CONFIDENCE = 0.5

model_path = askopenfilename(
    title="Select a YOLO model file",
    filetypes=[("PyTorch model", "*.pt")]
)
if not model_path:
    print("No model selected. Exiting...")
    exit()


image_path = askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)
if not image_path:
    print("No image selected. Exiting...")
    exit()


model = YOLO(model_path)
image = cv2.imread(image_path)
result = model(image, conf=CONFIDENCE)[0]

 
annotated_frame = result.plot()

plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
