from ultralytics import YOLO
import sys

model_path = 'models/emergency.pt'
try:
    m = YOLO(model_path)
    print('MODEL_LOADED', type(m))
    print('NAMES', getattr(m, 'names', None))
except Exception as e:
    print('MODEL_LOAD_ERROR', e)
    sys.exit(2)
