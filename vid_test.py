import random
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from ultralytics.utils import ops

model_path = "tile_detector.onnx"
model_classes = []
vid_path = "example_game2.mp4"
output_path = "output_video2.mp4"


class Detect:
    def __init__(self, model_path, ignore_classes=None, classes=None, input_size=(640, 640)):
        self.model_path = model_path
        self.classes = classes
        self.ignore_classes = ignore_classes if ignore_classes else []
        self.input_size = input_size
        self.model, self.device = self.load_model()

    def load_model(self):
        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in available_providers:
            onnx_provider = "CUDAExecutionProvider"
        elif "DmlExecutionProvider" in available_providers:
            onnx_provider = "DmlExecutionProvider"
        else:
            print("just, why ?")
            onnx_provider = "CPUExecutionProvider"

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = ort.InferenceSession(self.model_path, sess_options=so, providers=[onnx_provider])

        return model, onnx_provider

    def preprocess_image(self, img):
        # Ensure the image is a NumPy array
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Resize and pad image to the target size while preserving aspect ratio
        h, w, _ = img.shape
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a new image and pad it
        padded_img = np.full((self.input_size[1], self.input_size[0], 3), 128, dtype=np.uint8)
        padded_img[:new_h, :new_w, :] = resized_img

        # Convert BGR to RGB
        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

        # Normalize the image
        padded_img = padded_img.astype(np.float32) / 255.0

        # Reorder dimensions to (channels, height, width)
        padded_img = np.transpose(padded_img, (2, 0, 1))

        # Add the batch dimension
        padded_img = np.expand_dims(padded_img, axis=0)

        return torch.from_numpy(padded_img), new_w, new_h

    def postprocess(self, preds, img, orig_img_shape, resized_shape):
        # Apply Non-Maximum Suppression (NMS)
        preds = ops.non_max_suppression(
            preds,
            conf_thres=0.8,
            iou_thres=0.4,
            classes=None,
            agnostic=False,
        )

        orig_h, orig_w = orig_img_shape
        resized_w, resized_h = resized_shape

        # Calculate the scaling factor and padding
        scale_w = orig_w / resized_w
        scale_h = orig_h / resized_h

        results = []
        for pred in preds:
            if len(pred):
                pred[:, 0] *= scale_w  # x1
                pred[:, 1] *= scale_h  # y1
                pred[:, 2] *= scale_w  # x2
                pred[:, 3] *= scale_h  # y2
                results.append(pred.cpu().numpy())

        return results

    def detect_objects(self, img):
        # Convert PIL Image to NumPy array if it's not already a NumPy array
        if isinstance(img, Image.Image):
            img = np.array(img)
        # save image

        orig_h, orig_w = img.shape[:2]
        orig_img_shape = (orig_h, orig_w)

        # Preprocess the image
        preprocessed_img, resized_w, resized_h = self.preprocess_image(img)
        resized_shape = (resized_w, resized_h)

        # Run inference
        outputs = self.model.run(None, {'images': preprocessed_img.cpu().numpy()})

        # Postprocess the outputs
        detections = self.postprocess(torch.from_numpy(outputs[0]), preprocessed_img, orig_img_shape, resized_shape)

        results = {}
        for detection in detections:
            for *xyxy, conf, cls in detection:
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                class_name = self.classes[class_id]

                if class_id in self.ignore_classes or class_name in self.ignore_classes:
                    continue
                if class_name not in results:
                    results[class_name] = []
                results[class_name].append([x1, y1, x2, y2])

        return results


def draw_boxes(img, detections):
    colors = {
        x: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for x in model_classes
    }
    default_color = (0, 0, 0)
    for class_name, boxes in detections.items():
        color = colors.get(class_name, default_color)
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 4)
    return img


# Initialize detectors
main_detector = Detect(model_path, classes=model_classes)

cap = cv2.VideoCapture(vid_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

c = 0
s_time = time.time()

while cap.isOpened():
    if abs(s_time - time.time()) > 1:
        print(c, "IPS")
        s_time = time.time()
        c = 0
    ret, frame = cap.read()
    if not ret:
        break
    main_detections = main_detector.detect_objects(frame)

    frame = draw_boxes(frame, main_detections)
    out.write(frame)
    c += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
