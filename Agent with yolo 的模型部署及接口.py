# -*- coding: utf-8 -*-
#  Copyright (c) 2026 LRX
#  Author: LRX
#  File Name: agent_yolo_deploy.py
#  Project: Agent_with_yolo
#  Create Time: 2026-11-09
#  Description: YOLOv5 摄像头推理，检测结果+图像base64 POST发送至Agent后端接口

import cv2
import torch
import requests
import base64
import sys
import time

# 将 yolov5 目录加入模块搜索路径
sys.path.insert(0, "./yolov5")

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# ===================== CONFIG 配置项 =====================
WEIGHT = "./yolov5/weights/yolov5s.pt"   # 模型权重路径
AGENT_URL = "网址https" # Agent后端接收接口地址，改成你自己智能体服务实际 IP 和端口
CAM = 0                                  # 摄像头设备编号(默认)，如果摄像头读不到，可以尝试修改为 1、2；
CONF = 0.45                              # 目标置信度过滤阈值，值越高过滤越严格，只保留高置信度的目标。
# =========================================================


def to_b64(img):
    """OpenCV BGR图像转base64字符串"""
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    return ""


def send_data(pic_name: str, b64: str, res: list):
    """将图片base64与检测结果POST发送给Agent服务"""
    payload = {
        "pic_name": pic_name,
        "img": b64,
        "result": res,
        "status": "normal" if res else "no_target"
    }
    try:
        requests.post(AGENT_URL, json=payload, timeout=0.5)
    except Exception:
        print("[WARN] 发送失败，请检查Agent服务IP与端口是否正常")


def main():
    device = torch.device("cpu")
    # 加载YOLOv5模型
    model = DetectMultiBackend(WEIGHT, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size((640, 640), stride=stride)

    cap = cv2.VideoCapture(CAM)
    if not cap.isOpened():
        print("[ERROR] 摄像头打开失败，请检查设备编号与接线")
        return

    print(f"[INFO] 摄像头已启动，按 q 退出预览窗口")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取帧失败，退出")
            break

        pic_name = f"frame_{int(time.time() * 1000)}.jpg"

        # 预处理
        im = letterbox(frame, imgsz, stride=stride)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = torch.from_numpy(im).to(device).float() / 255.0
        im = im[None]

        # 推理 + NMS
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres=CONF)

        detect_result = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    detect_result.append({
                        "name": names[int(cls)],
                        "conf": float(conf),
                        "box": [int(x) for x in xyxy]
                    })

        b64_img = to_b64(frame)
        send_data(pic_name, b64_img, detect_result)

        # 绘制检测框预览
        for item in detect_result:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{item['name']} {item['conf']:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        cv2.imshow("preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()