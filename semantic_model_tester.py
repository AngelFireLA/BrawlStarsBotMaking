"""Preview PylaWallDetectorV3 semantic terrain on an image or image folder. See --help."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'models' / 'wall detection' / '.onnx' / 'PylaWallDetectorV3.onnx'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
COLORS_RGB = {
    'ground': (225, 207, 170), 'wall': (255, 139, 35), 'bush': (65, 175, 80),
    'rope_fence': (163, 72, 205), 'water': (35, 133, 245), 'out_of_bounds': (95, 60, 95),
    'indestructible_wall': (150, 25, 25), 'jump_pad': (255, 105, 200), 'teleport': (120, 70, 255),
    'heal_pad': (90, 255, 190), 'speed': (255, 245, 60), 'slow': (160, 225, 255),
    'damage': (175, 255, 20),
}
MIN_CONFIDENCE = {'wall': .60, 'bush': .60, 'rope_fence': .65, 'water': .75,
                  'out_of_bounds': .0, 'indestructible_wall': .60}


class SemanticDetector:
    """RGB letterbox -> ONNX class/confidence maps -> source-resolution uint8 class IDs."""

    def __init__(self, model_path=DEFAULT_MODEL, provider='cpu'):
        model_path = Path(model_path)
        manifest = json.loads(model_path.with_suffix('.json').read_text(encoding='utf-8'))
        self.names = [manifest['class_ids'][str(i)] for i in range(len(manifest['class_ids']))]
        self.height, self.width = manifest['height'], manifest['width']
        providers = {'cpu': 'CPUExecutionProvider', 'directml': 'DmlExecutionProvider',
                     'cuda': 'CUDAExecutionProvider'}
        selected = providers[provider]
        if selected not in ort.get_available_providers():
            raise ValueError(f'{selected} is not installed. Available: {ort.get_available_providers()}')
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        if provider == 'directml':
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=[selected] if provider == 'cpu'
                                            else [selected, 'CPUExecutionProvider'])
        self.provider = self.session.get_providers()[0]
        image_input = self.session.get_inputs()[0]
        if image_input.shape != [1, 3, self.height, self.width] or image_input.type != 'tensor(float)':
            raise ValueError('The model input does not match the manifest RGB float32 input size.')
        self.input_name = image_input.name
        outputs = {output.name for output in self.session.get_outputs()}
        if not {'class_ids', 'confidence'}.issubset(outputs):
            raise ValueError('Use the semantic deploy export with class_ids and confidence outputs.')

    def preprocess(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(self.height / h, self.width / w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (self.width - new_w) // 2, (self.height - new_h) // 2
        rgb = cv2.cvtColor(cv2.resize(bgr, (new_w, new_h)), cv2.COLOR_BGR2RGB)
        tensor = np.full((1, 3, self.height, self.width), np.float32(114 / 255), np.float32)
        tensor[0, :, top:top + new_h, left:left + new_w] = rgb.transpose(2, 0, 1).astype(np.float32) / 255
        return tensor, (left, top, new_w, new_h)

    def filter_classes(self, ids, confidence):
        """Reject weak pixels, then small/weak connected regions, on the unpadded model grid."""
        minimum = np.asarray([0 if name == 'ground' else MIN_CONFIDENCE.get(name, .50)
                              for name in self.names], np.float32)
        classes = np.where(confidence >= minimum[ids], ids, 0).astype(np.uint8)
        for class_id in np.unique(classes):
            if self.names[class_id] in {'ground', 'out_of_bounds'}:
                continue
            region = np.ascontiguousarray(classes == class_id, dtype=np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
            areas = stats[:, cv2.CC_STAT_AREA]
            means = np.bincount(labels.ravel(), weights=confidence.ravel(), minlength=count) / np.maximum(areas, 1)
            drop = (areas < .0006 * classes.size) | (means < .80)
            drop[0] = False
            classes[drop[labels]] = 0
        return classes

    def predict(self, bgr, raw=False):
        start = time.perf_counter()
        tensor, (left, top, new_w, new_h) = self.preprocess(bgr)
        ids, confidence = self.session.run(['class_ids', 'confidence'], {self.input_name: tensor})
        crop = (0, slice(top, top + new_h), slice(left, left + new_w))
        ids, confidence = ids[crop], confidence[crop]
        if ids.dtype != np.uint8 or ids.max() >= len(self.names) or confidence.shape != ids.shape:
            raise ValueError('Unexpected semantic output classes, shape, or dtype.')
        classes = ids if raw else self.filter_classes(ids, confidence)
        # Class IDs are discrete: nearest-neighbor resizing preserves them.
        classes = cv2.resize(classes, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return classes, (time.perf_counter() - start) * 1000


def make_preview(bgr, classes, names, alpha):
    palette = np.asarray([COLORS_RGB.get(name, (128, 128, 128)) for name in names], np.uint8)[:, ::-1]
    colored = palette[classes]
    overlay = bgr.copy()
    selected = classes != 0
    overlay[selected] = np.rint((1 - alpha) * bgr[selected] + alpha * colored[selected]).astype(np.uint8)
    h, w = bgr.shape[:2]
    scale = min(530 / w, 600 / h, 1)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    panels = []
    for image, title in [(bgr, 'Original'), (overlay, 'Semantic overlay'), (colored, 'Class map')]:
        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        header = np.full((32, size[0], 3), 25, np.uint8)
        cv2.putText(header, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
        panels.append(np.vstack([header, image]))
    preview = np.hstack(panels)
    columns = max(1, min(5, preview.shape[1] // 250))
    legend = np.full((math.ceil(len(names) / columns) * 27 + 28, preview.shape[1], 3), 25, np.uint8)
    for i, name in enumerate(names):
        x, y = i % columns * (preview.shape[1] // columns) + 8, i // columns * 27 + 8
        cv2.rectangle(legend, (x, y), (x + 14, y + 14), tuple(int(c) for c in palette[i]), -1)
        cv2.putText(legend, f'{i}: {name}', (x + 20, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(legend, 'Predicted ground footprints; colors and IDs follow the model manifest.',
                (8, legend.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, .40, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([preview, legend]), colored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Screenshot or folder of images (recursive).')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='ONNX model; its JSON manifest must be beside it.')
    parser.add_argument('--output', type=Path, default=ROOT / 'semantic_previews')
    parser.add_argument('--provider', choices=['cpu', 'directml', 'cuda'], default='cpu')
    parser.add_argument('--alpha', type=float, default=.45, help='Overlay opacity from 0 to 1.')
    parser.add_argument('--raw', action='store_true', help='Show raw argmax classes without confidence/component filtering.')
    parser.add_argument('--no-show', action='store_true', help='Save results without opening a preview window.')
    parser.add_argument('--limit', type=int, help='Maximum images to process from a folder.')
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    if not source.exists():
        parser.error(f'Source does not exist: {source}')
    if source.is_dir() and (output == source or source in output.parents):
        parser.error('--output must be outside the source folder so previews are not processed on the next run')
    images = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if source.is_dir() else [source]
    if not images or any(path.suffix.lower() not in IMAGE_SUFFIXES for path in images):
        parser.error('Source must contain supported image files (PNG, JPEG, BMP, WebP or TIFF).')
    images = images[:args.limit] if args.limit else images
    detector = SemanticDetector(args.model.expanduser().resolve(), args.provider)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(model=str(args.model.resolve()), provider=detector.provider, filtered=not args.raw,
                  class_ids=dict(enumerate(detector.names)), colors_rgb=COLORS_RGB, images=[])
    print(f'Model: {args.model}\nProvider: {detector.provider}\nOutput: {output}', flush=True)
    try:
        for index, path in enumerate(images, 1):
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f'Could not decode image: {path}')
            classes, milliseconds = detector.predict(bgr, raw=args.raw)
            preview, colored = make_preview(bgr, classes, detector.names, args.alpha)
            relative = path.relative_to(source) if source.is_dir() else Path(path.name)
            prefix = output / relative
            prefix.parent.mkdir(parents=True, exist_ok=True)
            for suffix, pixels in [('.preview.png', preview), ('.classes.png', classes), ('.colors.png', colored)]:
                if not cv2.imwrite(str(prefix) + suffix, pixels):
                    raise OSError(f'Could not write {prefix}{suffix}')
            coverage = np.bincount(classes.ravel(), minlength=len(detector.names)) / classes.size
            report['images'].append(dict(source=str(path), shape_hw=list(classes.shape),
                                         coverage=dict(zip(detector.names, coverage.tolist())),
                                         processing_ms=milliseconds, preview=str(prefix) + '.preview.png'))
            (output / 'predictions.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'[{index}/{len(images)}] {path.name}: {milliseconds:.1f} ms', flush=True)
            if not args.no_show:
                cv2.imshow('Pyla semantic model - press a key for the next image, Esc to stop', preview)
                if cv2.waitKey(0) & 0xFF == 27:
                    break
    finally:
        if not args.no_show:
            cv2.destroyAllWindows()
    print(f'Saved {len(report["images"])} predictions. Coverage is image area, not confidence or accuracy.')


if __name__ == '__main__':
    main()
