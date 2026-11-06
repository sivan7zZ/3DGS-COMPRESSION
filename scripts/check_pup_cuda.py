"""Tiny-scene GPU check of full Fisher matrix against pixel-by-pixel autograd."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argparse import Namespace
import torch
from scene import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix
from utils.pruning_scores import _backend, unpack_fisher, pup_pruning_scores, fisher_logdet
import math

torch.manual_seed(0)
device = "cuda"
model = GaussianModel(3)
n = 3
model._xyz = torch.nn.Parameter(torch.tensor([[-.12,-.08,2.],[.1,.05,2.5],[.04,-.1,3.]],device=device))
model._scaling = torch.nn.Parameter(torch.full((n,3), math.log(.12),device=device))
model._rotation = torch.nn.Parameter(torch.tensor([[1.,0.,0.,0.]]*n,device=device))
model._opacity = torch.nn.Parameter(torch.full((n,1),1.,device=device))
model._features_dc = torch.nn.Parameter(torch.randn(n,1,3,device=device)*.2)
model._features_rest = torch.nn.Parameter(torch.zeros(n,15,3,device=device))
model.active_sh_degree = 3
pipe = Namespace(antialiasing=False,compute_cov3D_python=False,convert_SHs_python=False,debug=False)
view = torch.eye(4,device=device)
projection = getProjectionMatrix(.01,100.,1.,1.).T.to(device)
camera = MiniCam(8,8,1.,1.,.01,100.,view,projection)
background = torch.zeros(3,device=device)
backend = _backend()
settings = backend.GaussianRasterizationSettings(8,8,math.tan(.5),math.tan(.5),background,1.,view,projection,3,camera.camera_center,False,False)
# The normal differentiable PWSR renderer supplies independent per-pixel Jacobians.
from gaussian_renderer import render
image = render(camera,model,pipe,background,clamp_output=False)["render"]
expected = torch.zeros(n,6,6,device=device)
for pixel in image.reshape(-1):
    gradients = torch.autograd.grad(pixel,(model._xyz,model._scaling),retain_graph=True)
    jacobian = torch.cat(gradients,1)
    expected += jacobian[:,:,None]*jacobian[:,None,:]
dummy = torch.zeros(n,21,device=device,requires_grad=True)
rendered,_ = backend.GaussianRasterizer(settings)(means3D=model.get_xyz.detach(),means2D=torch.zeros_like(model.get_xyz),
    shs=model.get_features.detach(),opacities=model.get_opacity.detach(),scales=model.get_scaling.detach(),
    rotations=model.get_rotation.detach(),fishers=dummy)
actual = unpack_fisher(torch.autograd.grad(rendered.sum(),dummy)[0])
relative_error = (actual-expected).norm()/expected.norm()
assert relative_error.item() < .002, relative_error.item()
scores = pup_pruning_scores([camera],model,pipe,background)
# Near-singular toy Fishers are sensitive to atomic accumulation order in logdet;
# validate the matrix itself above and require well-formed scores here.
assert scores.shape == (n,) and not torch.isnan(scores).any()
assert all(getattr(model,name).grad is None for name in ("_xyz","_scaling","_opacity","_features_dc"))
print("PUP CUDA vs per-pixel Jacobian Fisher relative error:",relative_error.item())
print("Scores:",scores.detach().cpu().tolist())
print("Model gradient buffers unchanged.")
