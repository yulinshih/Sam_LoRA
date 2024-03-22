from segment_anything import build_sam, SamAutomaticMaskGenerator 
from segment_anything import sam_model_registry
from sam_lora import LoRA_Sam
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_b"](checkpoint="/home/u6693411/SAM/SAM_ckpt/sam_vit_b_01ec64.pth")
sam.to(device)
print("sam_loaded")
lora_sam = LoRA_Sam(sam,r = 4)
lora_sam.to(device)
print("Lora_sam")
result = lora_sam.sam.image_encoder(torch.rand(size=(1,3,1024,1024)).to(device))
print(result.shape)