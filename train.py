from ultralytics import RTDETR
import os
import torch
import os
import torch
from thop import profile
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# Load a model
if __name__ == '__main__':


    model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-FasterNet.yaml")
    #model = RTDETR("last.pt")
    # 输出模型结构
    print("=== Model Architecture ===")
    print(model.model)

    print("\n=== Model Configuration ===")
    print(model.model.yaml)
    model.info()
    # Use the model
    device = next(model.model.parameters()).device
    input = torch.randn(1, 3, 640, 640).to(device)

    flops_load, params_load = profile(model.model, inputs=(input,), verbose=False)

    print(f"Load:  GFLOPs={flops_load / 1e9:.2f}, Params={params_load / 1e6:.2f}M")
    model.train(data="SpikeDataSet_biaozhun.yaml", cfg="ultralytics/cfg/default.yaml", epochs=100,imgsz=640, batch=4, workers=4, amp=False, device=[0],project='runs/detect',resume=False)  # train the model
