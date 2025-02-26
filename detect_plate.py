# Import library yang diperlukan
import argparse
import os
import sys
from pathlib import Path
import json
import threading
import torch
import cv2
import easyocr
import numpy as np
from websocket_server import WebsocketServer
import re

# Menentukan direktori root (folder utama) dari proyek
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Direktori root YOLO
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Tambahkan ROOT ke PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relatif terhadap direktori kerja

# Import fungsi dan kelas dari YOLOv5
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Konfigurasi WebSocket
WS_HOST = "0.0.0.0"  # Host WebSocket (bisa diakses dari mana saja)
WS_PORT = 9001  # Port WebSocket
ws_server = WebsocketServer(port=WS_PORT, host=WS_HOST)  # Buat server WebSocket

# Inisialisasi EasyOCR untuk membaca teks dari gambar (gunakan bahasa Inggris)
reader = easyocr.Reader(['en'], gpu=False)  # Gunakan CPU jika tidak ada GPU

def preprocess_plate(plate_img):
    """Preprocessing gambar plat nomor (ubah ke format RGB)"""
    return cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)  # Ubah warna dari BGR ke RGB

def extract_text_from_plate(plate_img):
    """Ekstrak teks dari gambar plat nomor menggunakan EasyOCR"""
    preprocessed = preprocess_plate(plate_img)  # Preprocessing gambar
    txt = reader.readtext(preprocessed, detail=0)  # Baca teks dari gambar
    raw_text = " ".join(txt)  # Gabungkan semua teks yang terdeteksi
    cleaned_text = re.sub(r'\s{2,}', ' ', raw_text)  # Hapus spasi berlebih
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', cleaned_text)  # Hapus simbol tidak valid
    split_text = cleaned_text.split()  # Pisahkan teks menjadi kata-kata

    # Format teks plat nomor (contoh: B 1234 ABC)
    if len(split_text) >= 3:
        split_text[2] = split_text[2][:3]  # Ambil 3 karakter pertama untuk bagian huruf
        processed_text = " ".join(split_text[:3])  # Gabungkan 3 bagian pertama
    else:
        processed_text = " ".join(split_text)  # Jika kurang dari 3 bagian, gabungkan semua

    # Cari pola plat nomor yang valid (contoh: B 1234 ABC)
    match = re.search(r'\b[A-Z]{1,2}\s\d{1,4}\s[A-Z]{1,3}\b', processed_text)
    if match:
        plate_number = match.group()  # Ambil teks yang sesuai dengan pola
        print("DETECTED PLATE NUMBER:", plate_number)
        return plate_number
    else:
        print("No valid plate number found in:", processed_text)
        return None

def send_data_to_ws(server, frame, plate_data):
    """Kirim data plat nomor dan frame ke WebSocket"""
    _, buffer = cv2.imencode('.jpg', frame)  # Ubah frame ke format JPEG
    frame_bytes = buffer.tobytes()  # Ubah frame ke byte
    plate_json = {
        "plates": plate_data,
        "frame": frame_bytes.hex()  # Ubah byte ke format hex untuk dikirim
    }
    server.send_message_to_all(json.dumps(plate_json))  # Kirim data ke semua klien WebSocket

def run(weights=ROOT / 'best.onnx', source='rtsp://admin:S3mangat45**@192.168.1.64',
        imgsz=(640, 640), conf_thres=0.5, iou_thres=0.45, device=''):
    """Fungsi utama untuk menjalankan deteksi plat nomor"""
    # Pilih perangkat (CPU atau GPU)
    device = select_device(device)
    # Muat model YOLO
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # Periksa ukuran gambar

    # Muat stream video (RTSP atau kamera)
    try:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    except Exception as e:
        print(f"Error loading stream: {e}")
        sys.exit(1)

    # Panaskan model (persiapan untuk inferensi)
    model.warmup(imgsz=(1, 3, *imgsz))

    # Buat direktori untuk menyimpan gambar plat nomor yang terdeteksi
    save_dir = Path('cropped_plates')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loop untuk memproses setiap frame dari stream video
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device).float() / 255.0  # Ubah frame ke format tensor
        if len(im.shape) == 3:
            im = im[None]  # Tambahkan dimensi batch jika diperlukan

        # Deteksi objek menggunakan model YOLO
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)  # Filter deteksi

        # Loop untuk setiap deteksi dalam frame
        for i, det in enumerate(pred):
            im0 = im0s[i].copy()  # Salin frame asli
            annotator = Annotator(im0, line_width=2, example=str(names))  # Siapkan annotator
            plates = []  # Simpan data plat nomor yang terdeteksi

            if len(det):  # Jika ada deteksi
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # Sesuaikan ukuran kotak deteksi

                for *xyxy, conf, cls in reversed(det):  # Loop untuk setiap objek yang terdeteksi
                    c = int(cls)  # Kelas objek
                    label = f'{names[c]} {conf:.2f}'  # Label dan confidence
                    
                    if names[c] == "License_Plate" and conf >= 0.6:  # Jika objek adalah plat nomor dan confidence >= 60%
                        x1, y1, x2, y2 = map(int, xyxy)  # Koordinat kotak deteksi
                        plate_crop = im0[y1:y2, x1:x2]  # Potong gambar plat nomor

                        if plate_crop.size == 0:  # Jika gambar kosong, lanjutkan
                            continue

                        plate_text = extract_text_from_plate(plate_crop)  # Ekstrak teks dari plat nomor
                        if plate_text:  # Jika teks valid
                            plate_filename = save_dir / f'plate_{i}.jpg'  # Nama file untuk menyimpan gambar
                            cv2.imwrite(str(plate_filename), plate_crop)  # Simpan gambar plat nomor
                            plates.append({
                                "text": plate_text,
                                "confidence": float(conf),
                                "bbox": [x1, y1, x2, y2]
                            })  # Simpan data plat nomor

                            annotator.box_label(xyxy, plate_text, color=colors(c, True))  # Gambar kotak dan teks di frame
            
            im0 = annotator.result()  # Hasil frame yang sudah di-annotate
            # Kirim frame ke WebSocket meskipun tidak ada plat nomor yang terdeteksi
            if not plates:
                plates = [{
                    "text": "No plate detected",
                    "confidence": 0.0,
                    "bbox": [0, 0, 0, 0]
                }]
            send_data_to_ws(ws_server, im0, plates)  # Kirim data ke WebSocket

def parse_opt():
    """Parse argumen dari command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'best.onnx', help='Path ke model YOLO')
    parser.add_argument('--source', type=str, default='rtsp://admin:S3mangat45**@192.168.1.64', help='URL kamera RTSP')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='Ukuran gambar untuk inferensi')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Threshold confidence untuk deteksi')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='Threshold IoU untuk NMS')
    parser.add_argument('--device', default='', help='Perangkat yang digunakan (CPU atau GPU)')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # Sesuaikan ukuran gambar
    return opt

def start_ws_server():
    """Jalankan server WebSocket"""
    print(f"WebSocket server started on ws://localhost:{WS_PORT}")
    try:
        ws_server.run_forever()  # Jalankan server WebSocket selamanya
    except Exception as e:
        print(f"WebSocket server error: {e}")

def main(opt):
    """Fungsi utama untuk menjalankan program"""
    run(**vars(opt))  # Jalankan fungsi run dengan argumen yang sudah di-parse

if __name__ == "__main__":
    # Jalankan server WebSocket di thread terpisah
    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()

    # Parse argumen dan jalankan program utama
    opt = parse_opt()
    main(opt)