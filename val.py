import os
from ultralytics import RTDETR
import cv2
import torch
from thop import profile

if __name__=="__main__":
	# Load a model
	model = RTDETR('best.pt')  # load a custom model
	# 输出模型结构
	print("=== Model Architecture ===")
	print(model.model)

	print("\n=== Model Configuration ===")
	print(model.model.yaml)
	model.info()
	device = next(model.model.parameters()).device
	input = torch.randn(1, 3, 640, 640).to(device)

	flops_load, params_load = profile(model.model, inputs=(input,), verbose=False)

	print(f"Load:  GFLOPs={flops_load / 1e9:.2f}, Params={params_load / 1e6:.2f}M")
	# Validate the model
	metrics = model.val(data='SpikeDataSet_test.yaml',plots=True,save=True,save_conf=False,save_txt=True,name='output', show_labels=False,show_conf=False,iou=0.7,conf=0.3,imgsz=640)   # no arguments needed, dataset and settings remembered
	#model.conf

	# metrics.box.map    # map50-95
	# metrics.box.map50  # map50
	# metrics.box.map75  # map75
	# metrics.box.maps   # a list contains map50-95 of each category
	metrics = model.predict(source='demo_data/images/test', imgsz=640, iou=0.5, conf=0.3,
							save=True, show_labels=False,save_txt=True, show_conf=False, line_width=10, max_det=1000)
