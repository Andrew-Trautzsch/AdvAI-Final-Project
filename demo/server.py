import os
import tempfile
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_file

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

import cv2

app = Flask(__name__, static_folder='.', static_url_path='')

# Configure a YOLO model path. Set YOLO_MODEL_PATH to override.
MODEL_PATH = os.environ.get('YOLO_MODEL_PATH') or str(Path(__file__).resolve().parent / 'yolo11x_visdrone_50epoch.pt')
DEFAULT_FALLBACK_MODEL = 'yolov8n.pt'


def load_model():
    if YOLO is None:
        raise RuntimeError('Missing ultralytics package. Install with `pip install ultralytics`.')

    model_path = MODEL_PATH
    if not Path(model_path).exists():
        app.logger.warning('Model not found at %s, falling back to %s', model_path, DEFAULT_FALLBACK_MODEL)
        model_path = DEFAULT_FALLBACK_MODEL

    return YOLO(model_path)


try:
    model = load_model()
    app.logger.info('Loaded YOLO model from %s', MODEL_PATH)
except Exception as exc:
    model = None
    app.logger.warning('YOLO model not loaded: %s', exc)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/process', methods=['POST'])
def process_video():
    if model is None:
        return jsonify({'error': 'YOLO model is not available. Check server logs and install dependencies.'}), 500

    if 'video' not in request.files:
        return jsonify({'error': 'Missing "video" file field.'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / 'input.mp4'
            output_path = Path(temp_dir) / 'processed.mp4'
            video_file.save(str(input_path))

            process_uploaded_video(str(input_path), str(output_path))

            return send_file(str(output_path), mimetype='video/mp4')
    except Exception as exc:
        app.logger.error('Failed to process video: %s', traceback.format_exc())
        return jsonify({'error': 'Video processing failed.', 'detail': str(exc)}), 500


def process_uploaded_video(input_video_path: str, output_video_path: str, conf_threshold: float = 0.25, img_size: int = 640):
    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise RuntimeError('Failed to open uploaded video file.')

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('Failed to initialize video writer.')

    while True:
        success, frame = capture.read()
        if not success:
            break

        result = model(frame, imgsz=img_size, conf=conf_threshold)
        annotated_frame = draw_boxes(frame, result[0])
        writer.write(annotated_frame)

    capture.release()
    writer.release()


def draw_boxes(frame, result):
    annotated = frame.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    names = getattr(model, 'names', {}) or {}
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else []
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
    classes = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else []

    for box, conf, cls in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names.get(cls, cls)} {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
