# =================================\CONFIG./=====================================
# default System camera access code = 0, if connect with any external camera put 1,2 and so on with the no of connected cameras.
CAMERA_NO = 0

# To count the total number of people (True/False).
PEOPLE_COUNTER = True

# Set if GPU should be used for computations; Otherwise uses the CPU by default.
USE_GPU = True

DETECTION_THRESHOLD = 0.6

NMS_THRESHOLD = 0.52

# Kaggle dataset identifier
DATASET_IDENTIFIER = 'shantanu1118/face-mask-detection-dataset-with-4k-samples'

# Different paths used:
PATH = {
    'DATASET': './dataset',
    'YOLO_LABELS': './yolo-coco/coco.names',
    'YOLO_WEIGHTS': './yolo-coco/yolov3.weights',
    'YOLO_CONFIG': './yolo-coco/yolov3.cfg',
    'SAMPLE_DATASET': './sample_dataset'
}

# ===============================================================================
