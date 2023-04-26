import os
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch
try:
    from simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
except ImportError:
    from .simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
import clip

dir_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

clip_model_name = 'ViT-B/16'
clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
clip_model.eval().requires_grad_(False)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

# 512 is embed dimension for ViT-B/16 CLIP
model = AestheticMeanPredictionLinearModel(512)
model.load_state_dict(
    torch.load("{}/models/sac_public_2022_06_29_vit_b_16_linear.pth".format(dir_path))
)
model = model.to(device)

def get_filepaths(parentpath, filepaths):
    paths = []
    for path in filepaths:
        try:
            new_parent = os.path.join(parentpath, path)
            paths += get_filepaths(new_parent, os.listdir(new_parent))
        except NotADirectoryError:
            paths.append(os.path.join(parentpath, path))
    return paths

def run(img):
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert('RGB')
    img = TF.resize(img, 224, transforms.InterpolationMode.LANCZOS)
    img = TF.center_crop(img, (224,224))
    img = TF.to_tensor(img).to(device)
    img = normalize(img)
    clip_image_embed = F.normalize(
        clip_model.encode_image(img[None, ...]).float(),
        dim=-1)
    score = model(clip_image_embed)
    if device != 'cpu':
        score = score.cpu()
    return score.item()
