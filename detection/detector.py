import os
import sys
import time
import logging
import re
from pathlib import Path

import cv2
import backports.lzma as lzma
sys.modules['lzma'] = lzma
import torch
import easyocr
import numpy as np

from config.config import Config
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

class ALPRSystem:
    def __init__(self):
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        self.cfg = Config()
        self.device = select_device(self.cfg.DEVICE)
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.model = DetectMultiBackend(self.cfg.WEIGHTS_PATH, device=self.device)
        self.model.warmup(imgsz=(1, 3, *self.cfg.IMG_SIZE))
        self.save_dir = self.cfg.SAVE_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.last_detected_plate = {
            "plate": None,
            "timestamp": 0
        }

    def preprocess_plate(self, plate_img):
        return cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

    def extract_text_from_plate(self, plate_img):
        preprocessed = self.preprocess_plate(plate_img)
        txt = self.reader.readtext(preprocessed, detail=0)
        raw_text = " ".join(txt)
        cleaned_text = re.sub(r'\s{2,}', ' ', raw_text)
        cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', cleaned_text)
        split_text = cleaned_text.split()

        if len(split_text) >= 3:
            split_text[2] = split_text[2][:3]
            processed_text = " ".join(split_text[:3])
        else:
            processed_text = " ".join(split_text)

        match = re.search(r'\b[A-Z]{1,2}\s\d{1,4}\s[A-Z]{1,3}\b', processed_text)
        if match:
            plate_number = match.group()
            current_time = time.time()

            if (self.last_detected_plate["plate"] == plate_number and
                    current_time - self.last_detected_plate["timestamp"] <= 5):
                logging.info(f"Duplicate plate detected within 5 seconds: {plate_number}")
                return None

            self.last_detected_plate["plate"] = plate_number
            self.last_detected_plate["timestamp"] = current_time

            logging.info(f"DETECTED PLATE NUMBER: {plate_number}")
            return plate_number
        else:
            logging.warning(f"No valid plate number found in: {processed_text}")
            return None

    def run(self):
        imgsz = self.cfg.IMG_SIZE
        conf_thres = self.cfg.CONF_THRESHOLD
        iou_thres = self.cfg.IOU_THRESHOLD
        show = self.cfg.SHOW_WINDOW
        source = self.cfg.RTSP_URL

        try:
            dataset = LoadStreams(source, img_size=imgsz, stride=self.model.stride, auto=self.model.pt)
        except Exception as e:
            logging.error(f"Error loading stream: {e}")
            sys.exit(1)

        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device).float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            pred = self.model(im)
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            for i, det in enumerate(pred):
                im0 = im0s[i].copy()
                annotator = Annotator(im0, line_width=2, example=str(self.model.names))

                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = f'{self.model.names[c]} {conf:.2f}'

                        if self.model.names[c] == "License_Plate" and conf >= 0.6:
                            x1, y1, x2, y2 = map(int, xyxy)
                            plate_crop = im0[y1:y2, x1:x2]

                            if plate_crop.size == 0:
                                continue

                            plate_text = self.extract_text_from_plate(plate_crop)
                            if plate_text:
                                timestamp = int(time.time() * 1000)
                                plate_filename = self.save_dir / f'plate_{timestamp}.jpg'
                                cv2.imwrite(str(plate_filename), plate_crop)
                                annotator.box_label(xyxy, plate_text, color=colors(c, True))

                im0 = annotator.result()
                if show:
                    cv2.imshow("LTMS ALPR", im0)
                    if cv2.waitKey(1) == ord('q'):
                        logging.info("Exiting by key press...")
                        break