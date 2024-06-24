import torch
import torchvision.models as models
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram
from torch import nn
from executorch.exir import EdgeProgramManager, to_edge, ExecutorchBackendConfig, ExecutorchProgramManager


NUM_CLASSES = 10

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=NUM_CLASSES)

# load weight trained by lab0
model_state_dict = torch.load("model.pt", map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)
model.eval()

example_args = (torch.randn(1, 3, 224, 224),)

pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_args)
print("Pre-Autograd ATen Dialect Graph")
print(pre_autograd_aten_dialect)

aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, example_args)
print("ATen Dialect Graph")
print(aten_dialect)

edge_program: EdgeProgramManager = to_edge(aten_dialect)
print("Edge Dialect Graph")
print(edge_program.exported_program())

executorch_program: ExecutorchProgramManager = edge_program.to_executorch(
    ExecutorchBackendConfig(
        passes=[],  # User-defined passes
    )
)

with open("mobilenet.pte", "wb") as file:
    file.write(executorch_program.buffer)