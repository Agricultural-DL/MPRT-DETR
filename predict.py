from ultralytics import RTDETR
import cv2


if __name__=="__main__":
	# Load a model
	#model = YOLO('yolov5l.pt')  # load an official model
	model = RTDETR('best.pt')  # load a custom model

	# Validate the model
	metrics = model.predict(source = 'demo_data/images/test',save=True,save_conf=True,save_txt=True,name='output',line_thickness=3, show_labels=False,show_conf=False,iou=0.5,conf=0.3,imgsz=640)  # no arguments needed, dataset and settings remembered
	# metrics.box.map    # map50-95
	# metrics.box.map50  # map50
	# metrics.box.map75  # map75
	# metrics.box.maps   # a list contains map50-95 of each category
