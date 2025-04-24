import os
from pathlib import Path
import torch
import logging

class Config:
    # Base directory (root of the project)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Model weights
    WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', str(BASE_DIR / 'models' / 'best.pt'))

    # Video Source
    SOURCE_TYPE = os.getenv('SOURCE_TYPE', 'rtsp')  # 'rtsp' or 'camera'
    RTSP_URL = os.getenv('RTSP_URL', 'rtsp://admin:S3mangat45**@192.168.1.64')
    CAMERA_ID = int(os.getenv('CAMERA_ID', 0))

    # Inference Settings
    IMG_SIZE = tuple(map(int, os.getenv('IMG_SIZE', '640,640').split(',')))
    CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', 0.5))
    IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.45))
    DEVICE = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'

    # Display Settings
    SHOW_WINDOW = bool(int(os.getenv('SHOW_WINDOW', 0)))  # 1 to show, 0 to hide

    # Save directory for cropped plates
    SAVE_DIR = BASE_DIR / 'data' / 'cropped_plates'
    
    # Logging
    LOG_DIR = BASE_DIR / 'logs'
    LOG_FILE = LOG_DIR / 'alpr.log'

    @classmethod
    def init(cls):
        cls.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()  # Optional: also log to console
            ]
        )

        logging.info("Configuration initialized.")