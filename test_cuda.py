import torch
from torchvision.ops import nms

boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 9, 9]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
iou_threshold = 0.5

try:
    indices = nms(boxes, scores, iou_threshold)
    print("NMS result:", indices)
except Exception as e:
    print("Error during NMS:", e)
