import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize
from torch.utils.data import Dataset, DataLoader, random_split
import json
import cv2
import os
import numpy as np
import pycocotools.mask as mask_utils
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Optional, Tuple
import random
# import SAM model
from segment_anything import SamPredictor, sam_model_registry
from transforms import ResizeLongestSide
import time
from segment_anything import build_sam, SamAutomaticMaskGenerator 
from sam_lora import LoRA_Sam
from transforms import ResizeLongestSide
'''
TO SETUP:
1. use box prompt?
2. use point prompt?
3. if box prompt, amodal or modal box
4. vit_b or vit_h
5. dataset name, kins or others
6. train_val_ratio
7. total training epoch number
8. learning rate
9. root for pre-computed img embeddings
10. path for training annotation
11. root for saving tensorboard writer
12. root for saving checkpoint
13. probability for bounding box augmentation
14. bounding box augmentation parameters
'''
is_box = True
is_point = False
box_type = 'modal'
vit_type = 'vit_b'
dataset_name = 'kins'
train_val_ratio = [0.8, 0.2]
EPOCHS = 2000
lr = 1e-4
img_root = '/work/u6693411/amodal_dataset/kins/training/training/image_2'
anno_path = '/work/u6693411/amodal_dataset/kins/KITTI_AMODAL_DATASET/train_dict_anno.json'
tb_save_path = '/work/u6693411/amodal_dataset/checkpoint/runs/'
ckpt_save_path = '/work/u6693411/amodal_dataset/checkpoint/'
box_aug_prob = 0.7
box_aug_param = ((0.1, 0.4), (0.7, 1.3))


vit_dict = {'vit_b':"sam_vit_b_01ec64.pth", 'vit_h':"sam_vit_h_4b8939.pth"}
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
sam = sam_model_registry[vit_type](checkpoint="/home/u6693411/SAM/SAM_ckpt/sam_vit_b_01ec64.pth")
sam.to(device)
lora_sam = LoRA_Sam(sam,r = 4)
lora_sam.to(device)
transform = ResizeLongestSide(sam.image_encoder.img_size)

@torch.no_grad()
def set_torch_image(
    transformed_image: torch.Tensor,
    original_image_size: Tuple[int, ...],
    ) -> None:
    """
    Calculates the image embeddings for the provided image, allowing
    masks to be predicted with the 'predict' method. Expects the input
    image to be already transformed to the format expected by the model.

    Arguments:
        transformed_image (torch.Tensor): The input image, with shape
        1x3xHxW, which has been transformed with ResizeLongestSide.
        original_image_size (tuple(int, int)): The size of the image
        before transformation, in (H, W) format.
    """
    assert (
        len(transformed_image.shape) == 4
        and transformed_image.shape[1] == 3
        and max(*transformed_image.shape[2:]) == sam.image_encoder.img_size
    ), f"set_torch_image input must be BCHW with long side {sam.image_encoder.img_size}."

    input_size = tuple(transformed_image.shape[-2:])
    input_image = sam.preprocess(transformed_image)
    #features = lora_sam.sam.image_encoder(input_image)
    return input_size, input_image


def set_image(
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
    """
    Calculates the image embeddings for the provided image, allowing
    masks to be predicted with the 'predict' method.

    Arguments:
        image (np.ndarray): The image for calculating masks. Expects an
        image in HWC uint8 format, with pixel values in [0, 255].
        image_format (str): The color format of the image, in ['RGB', 'BGR'].
    """
    assert image_format in [
        "RGB",
        "BGR",
    ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
    if image_format != sam.image_format:
        image = image[..., ::-1]

    # Transform the image to the form expected by the model
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    return set_torch_image(input_image_torch, image.shape[:2])

# polygon to binary mask
def polys_to_mask(polygons, height, width):
	rles = mask_utils.frPyObjects(polygons, height, width)
	rle = mask_utils.merge(rles)
	mask = mask_utils.decode(rle)
	return mask

def box_augment(box, range_TL, range_WH):
    # box with shape (N, 4): [[x y w h]..]
    # range_TL [0.1, 0.4]
    # range_WH [0.7, 1.3]
    N = box.shape[0]
    low, high = range_WH
    sampled_WH = np.random.uniform(low, high, (N, 2))
    random_WH = sampled_WH * box[:, 2:]

    low, high = range_TL
    sampled_TL_WH = np.random.uniform(low, high, (N, 2))
    sampled_TL_WH *= box[:, 2:]
    TL_tl = box[:, :2] - sampled_TL_WH / 2
    TL_br = box[:, :2] + sampled_TL_WH / 2
    sizes = TL_br - TL_tl
    random_offsets = np.random.rand(N, 2) * sizes
    random_points = TL_tl + random_offsets
    return np.concatenate((random_points, random_WH), axis=1)

# randomly pick points for point prompt
def pick_rand_point(isegm, asegm, abbox, raw=False):
    N, C, H, W = asegm.shape
    amask_tmp = torch.ones((N, C, H, W), dtype=asegm.dtype, device=asegm.device)
    abbox = np.hstack((abbox[:, :2], abbox[:, :2] + abbox[:, 2:]))
    abbox[:, [0, 1]] = abbox[:, [0, 1]] - 10
    abbox[:, [2, 3]] = abbox[:, [2, 3]] + 10
    abbox[:, [0, 2]] = np.clip(abbox[:, [0, 2]], 0, W)
    abbox[:, [1, 3]] = np.clip(abbox[:, [1, 3]], 0, H)
    abbox = np.hstack((abbox[:, :2], abbox[:, 2:] - abbox[:, :2]))
    for i in range(N):
        x, y, w, h = abbox[i]
        amask_tmp[i, :, y:y+h, x:x+w] = asegm[i, :, y:y+h, x:x+w]
    
    cor_list = []
    lab_list = []

    for i in range(N):
        # Identifying foreground and background points
        fore_p = (isegm[i, 0] == 1).nonzero(as_tuple=False)
        back_p = (amask_tmp[i, 0] == 0).nonzero(as_tuple=False)
        if raw:
            back_p = (asegm[i, 0] == 0).nonzero(as_tuple=False)

        if fore_p.nelement() == 0:
            print("no foreground point")
            indices = torch.randperm(back_p.shape[0])[:2]
            cor = back_p[indices]
            lab_list.append(torch.tensor([0, 0]))
        else:
            indices = torch.tensor([torch.randint(0, len(fore_p), (1,)), torch.randint(0, len(back_p), (1,))]).view(-1)
            fore_p_selected = fore_p[indices[0]]
            back_p_selected = back_p[indices[1]]
            cor = torch.stack((fore_p_selected, back_p_selected))
            lab_list.append(torch.tensor([1, 0]))
        
        # Adjusting coordinates order from (x, y) to (y, x) for consistency
        cor = cor[:, [1, 0]]
        cor_list.append(cor)

    cor_tensor = torch.stack(cor_list)
    lab_tensor = torch.stack(lab_list)
    return cor_tensor, lab_tensor

class AmodalDataset(Dataset):
    def __init__(self, dataset='kins', box_type='modal'):
        self.dataset = dataset
        self.cur_imgid = None
        self.cur_imgemb = None
        self.box_type = box_type
        if dataset == 'kins':
            self.data_root = img_root
            self.anno_path = anno_path
            with open(self.anno_path) as f:
                anns = json.load(f)
                self.imgs_info = anns['images']
                self.anns_info = anns["annotations"]
    def __getitem__(self, index):
        img_id = self.imgs_info[index]["id"]
        img_name = self.imgs_info[index]["file_name"]
        path = os.path.join(self.data_root, img_name)
        
        # load pre-computed img embeddings
        img = cv2.imread(os.path.join(self.data_root, img_name), cv2.IMREAD_COLOR)
        original_size = img.shape[:2]
        input_size, input_image = set_image(img, image_format='BGR')
        h, w = original_size

        # amodal mask GT
        asegm = [anno["a_segm"] for anno in self.anns_info[str(img_id)]]
        asegm = np.stack([polys_to_mask(mask, h, w) for mask in asegm])
        asegm = torch.as_tensor(asegm, dtype=torch.float, device=device).unsqueeze(1)  

        # amodal bbox GT
        abbox = [anno["a_bbox"] for anno in self.anns_info[str(img_id)]]

        point_torch = []
        if is_point:
            # inmodal mask GT
            isegm = [anno["i_segm"] for anno in self.anns_info[str(img_id)]]
            isegm = np.stack([polys_to_mask(mask, h, w) for mask in isegm])
            isegm = torch.as_tensor(isegm, dtype=torch.float, device=device).unsqueeze(1)
            cor_tensor, lab_tensor = pick_rand_point(isegm, asegm, np.array(abbox))
            cor_tensor = transform.apply_coords_torch(cor_tensor, original_size)
            point_torch = (cor_tensor, lab_tensor)

        box_torch = []
        if is_box:
            # inmodal / amodal bounding box GT for prompt
            bbox = abbox
            if self.box_type != 'amodal':
                bbox = [anno["i_bbox"] for anno in self.anns_info[str(img_id)]]
            
            # perform bbox augmentation with set probability 'box_aug_prob'
            if np.random.random() <= box_aug_prob:
                bbox_aug = box_augment(np.array(bbox), box_aug_param[0], box_aug_param[1])
            else:
                bbox_aug = np.array(bbox)
            box_torch = np.hstack((bbox_aug[:, :2], bbox_aug[:, :2] + bbox_aug[:, 2:]))
            box_torch = transform.apply_boxes(box_torch, original_size)
            box_torch = torch.as_tensor(box_torch, dtype=torch.float, device=device)
        return input_image, asegm, box_torch, point_torch, original_size, input_size

    def __len__(self):
        return len(self.imgs_info)


optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr = lr)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Create datasets for training & validation
dataset = AmodalDataset(dataset_name, box_type)
training_set, validation_set = random_split(dataset, train_val_ratio)

# shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        data = [None if x == [] else x for x in data]
        input_image, asegm, bbox, point, original_size, input_size = data
        input_image = torch.squeeze(input_image, 0)
        image_embedding = lora_sam.sam.image_encoder(input_image)
        asegm = torch.squeeze(asegm, 0)
        if is_box:
            bbox = torch.squeeze(bbox, 0)
        if is_point:
            point = (point[0].squeeze(0), point[1].squeeze(0))
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=point,
                boxes=bbox,
                masks=None,
            )
            
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        upscaled_masks = sam.postprocess_masks(low_res_masks, input_size, original_size).to(device)
        
        # Compute the loss and its gradients
        loss = loss_fn(upscaled_masks, asegm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    
    return last_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(os.path.join(tb_save_path, 'fashion_trainer_{}'.format(timestamp)))
epoch_number = 0

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # gradient tracking is on
    sam.mask_decoder.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    sam.mask_decoder.eval()

    # reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vdata = [None if x == [] else x for x in vdata]
            vinput_image, vasegm, vbbox, vpoint, voriginal_size, vinput_size = vdata
            vinput_image = torch.squeeze(vinput_image, 0)
            vimage_embedding = lora_sam.sam.image_encoder(vinput_image)
            vasegm = torch.squeeze(vasegm, 0)
            if is_box:
                vbbox = torch.squeeze(vbbox, 0)
            if is_point:
                vpoint = (vpoint[0].squeeze(0), vpoint[1].squeeze(0))
            vsparse_embeddings, vdense_embeddings = sam.prompt_encoder(
                points=vpoint,
                boxes=vbbox,
                masks=None,
            )
            
            vlow_res_masks, viou_predictions = sam.mask_decoder(
                image_embeddings=vimage_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=vsparse_embeddings,
                dense_prompt_embeddings=vdense_embeddings,
                multimask_output=False,
            )

            vupscaled_masks = sam.postprocess_masks(vlow_res_masks, vinput_size, voriginal_size).to(device)
            vloss = loss_fn(vupscaled_masks, vasegm)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = os.path.join(ckpt_save_path, 'model_{}_{}'.format(timestamp, epoch_number))
        torch.save(sam.mask_decoder.state_dict(), model_path)

    epoch_number += 1
