import argparse
import os
import sys
from pathlib import Path
import json
from websocket_server import WebsocketServer
import threading
import torch
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

WS_HOST = "0.0.0.0"
WS_PORT = 9001
ws_server = WebsocketServer(port=WS_PORT, host=WS_HOST)

def send_data_to_ws(server, frame, plate_data):
    _, buffer = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
    frame_bytes = buffer.tobytes()
    plate_json = {
        "plates": plate_data,
        "frame": frame_bytes.hex()  # Convert to hex for transmission
    }
    server.send_message_to_all(json.dumps(plate_json))

def run(weights=ROOT / 'best.onnx', source='rtsp://admin:S3mangat45**@192.168.1.64', imgsz=(300, 300), conf_thres=0.25, iou_thres=0.45, device='', view_img=False):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0  # uint8 to fp32 and normalize
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):  # per image
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=2, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                save_dir = Path('cropped_plates')
                save_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

                seen = 0

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    print(label)
                    
                    if names[c] == "License_Plate" and conf >= 0.6:
                        plate = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        
                        plate_filename = save_dir / f'plate_{seen}_{i}.jpg'
                        cv2.imwrite(str(plate_filename), plate)
                        print(f"Saved cropped license plate to {plate_filename}")
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
                seen += 1
                
            im0 = annotator.result()
            send_data_to_ws(ws_server, im0, "test")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'best.onnx', help='model path')
    parser.add_argument('--source', type=str, default='rtsp://admin:S3mangat45**@192.168.1.64', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def start_ws_server():
    print(f"WebSocket server started on ws://localhost:{WS_PORT}")
    ws_server.run_forever()

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    ws_thread = threading.Thread(target=start_ws_server)
    ws_thread.daemon = True
    ws_thread.start()
    opt = parse_opt()
    main(opt)