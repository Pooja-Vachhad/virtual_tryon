import torch
import numpy as np
import onnxruntime as ort
from .model import MobileNet

weights = "best.pth"
onnx_file = "lip_seg.onnx"

#loading the model
model = MobileNet(num_classes = 1, pretrained = False)
model.load_state_dict(torch.load(weights , map_location= "cpu"))
model.eval()


# example input, used to trace the model
dummy = torch.randn(1 , 3 , 256 , 256)

#export the pth -> onnx format
torch.onnx.export(model , dummy , onnx_file , export_params = True, opset_version=18 ,
                  do_constant_folding= True,
                  input_names = ["image"],
                  output_names = ["logits"],
                  dynamic_axes = {"image": {0:"batch"} , "logits":{0:"batch"}},
                  )

print("saved", onnx_file)

# check that ONNX gives the same output as PyTorch
sess = ort.InferenceSession(onnx_file , providers=["CPUExecutionProvider"])
x = torch.randn(2 , 3 , 256 , 256)
with torch.no_grad():
   torch_out = model(x).numpy()
onnx_out = sess.run(None , {"image":x.numpy()})[0]



print("output shape:", onnx_out.shape)
print("max difference:", np.abs(torch_out - onnx_out).max())