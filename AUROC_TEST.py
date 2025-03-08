import argparse
import importlib
import matplotlib
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage import morphology
from scipy.ndimage import gaussian_filter
import imageio
from sklearn import datasets
from utils.metric import *
import numpy as np
import matplotlib.pyplot as plt
from utils.model_helper import ModelHelper
from utils.utils import *
from efficientnet_pytorch import EfficientNet
from easydict import EasyDict
import yaml
import ipdb

import datasets.MIP as mip
import datasets.LEGO_3D as lego
import datasets.MIPreal as real

def plot_fig(test_img, recon_imgs, test_imgs_R, recon_imgs_R, scores, gts, threshold, save_dir,class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2*threshold*255)
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        img_R = test_imgs_R[i]
        img_R = denormalization(img_R)
        recon_img = recon_imgs[i]
        recon_img = denormalization(recon_img)
        recon_img_R = recon_imgs_R[i]
        recon_img_R = denormalization(recon_img_R)
        gt = gts[i]
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(2)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 6, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(recon_img)
        ax_img[1].title.set_text('Reconst')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', norm=norm)
        ax_img[3].title.set_text('Predicted heat map')
        ax_img[4].imshow(mask, cmap='gray')
        ax_img[4].title.set_text('Predicted mask')
        ax_img[5].imshow(vis_img)
        ax_img[5].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}_png'.format(i)), dpi=300)
        plt.close()

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
from skimage import morphology
from skimage.segmentation import mark_boundaries

#def denormalization(img):
    # Assuming img is normalized between 0 and 1
#    return img * 255

def plot_fig2(test_img, recon_imgs, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2 * threshold * 255)
    
    for i in range(num):
        img = denormalization(test_img[i])
        recon_img = denormalization(recon_imgs[i])
        gt = gts[i]
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(2)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        
        fig_img, ax_img = plt.subplots(1, 6, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        for ax_i in ax_img:
            ax_i.axis('off') 
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(recon_img)
        ax_img[1].title.set_text('Reconst')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', norm=norm)
        ax_img[3].title.set_text('Predicted heat map')
        ax_img[4].imshow(mask, cmap='gray')
        ax_img[4].title.set_text('Predicted mask')
        ax_img[5].imshow(vis_img)
        ax_img[5].title.set_text('Segmentation result')

        
        fig_img.savefig(os.path.join(save_dir, f"{class_name}_{i}_png"), dpi=300)
        
        
        # single_image_names = ['Image', 'Reconst', 'GroundTruth', 'Predicted_heat_map', 'Predicted_mask', 'Segmentation_result']
        # for j, ax in enumerate(ax_img):
        #     single_fig, single_ax = plt.subplots(figsize=(4, 4))
        #     single_ax.imshow(ax.images[0].get_array(), cmap=ax.images[0].cmap if ax.images[0].cmap else None, norm=norm if j==3 else None)
        #     single_ax.axis('off')  
        #     single_fig.savefig(os.path.join(single_save_dir, f"{class_name}_{i}_{single_image_names[j]}.png"), dpi=100, bbox_inches='tight', pad_inches=0)
        #     plt.close(single_fig)  

        # plt.close(fig_img) 

def plot_fig_all(test_img, recon_imgs, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2 * threshold * 255)
    
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        recon_img = recon_imgs[i]
        recon_img = denormalization(recon_img)
        gt = gts[i]
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(2)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        #vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        
        # Plot individual images
        titles = ['Image', 'Reconst', 'GroundTruth', 'Predicted heat map', 'Predicted mask']#, 'Segmentation result']
        images = [img, recon_img, gt, heat_map, mask]#, vis_img]
        
        for idx, (title, image) in enumerate(zip(titles, images)):
            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
            ax.imshow(image, cmap='gray' if idx < 3 else 'jet', norm=norm if idx == 3 else None)
            if idx == 3:
                ax.imshow(img, cmap='gray', interpolation='none')
                ax.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', norm=norm)
            # ax.set_title(title)
            ax.axis('off')  # Remove axes
            
            """if idx == 3:  # Add colorbar only for heat map
                left = 0.92
                bottom = 0.15
                width = 0.015
                height = 1 - 2 * bottom
                rect = [left, bottom, width, height]
                cbar_ax = fig.add_axes(rect)
                cb = plt.colorbar(ax.images[0], cax=cbar_ax, fraction=0.046, shrink=0.6)
                cb.ax.tick_params(labelsize=8)
                cb.set_label('Anomaly Score', fontsize=8)"""
            
            file_path = os.path.join(save_dir, f'{class_name}_{i}_{idx}.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        
def update_config(config):
    # update feature size
    _, reconstruction_type = config.net[2].type.rsplit(".", 1)
    if reconstruction_type == "UniAD":
        input_size = config.dataset.input_size
        outstride = config.net[1].kwargs.outstrides[0]
        assert (
            input_size[0] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        assert (
            input_size[1] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        feature_size = [s // outstride for s in input_size]
        config.net[2].kwargs.feature_size = feature_size

    # update planes & strides
    backbone_path, backbone_type = config.net[0].type.rsplit(".", 1)
    module = importlib.import_module(backbone_path)
    backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outblocks = None
    if "efficientnet" in backbone_type:
        outblocks = []
    outstrides = []
    outplanes = []
    for layer in config.net[0].kwargs.outlayers:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        if "efficientnet" in backbone_type:
            outblocks.append(backbone["blocks"][idx])
        outstrides.append(backbone["strides"][idx])
        outplanes.append(backbone["planes"][idx])
    if "efficientnet" in backbone_type:
        config.net[0].kwargs.pop("outlayers")
        config.net[0].kwargs.outblocks = outblocks
    config.net[0].kwargs.outstrides = outstrides
    config.net[1].kwargs.outplanes = [sum(outplanes)]

    return config

with open("retrieval/config.yaml") as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
config = update_config(config)
model = ModelHelper(config.net)
model.eval()
model.cuda()


def compare_feature(ref_feature,rgb_feature):
    loss=(ref_feature-rgb_feature)**2 
    result_sum=torch.mean(loss,axis=0) 
    result_sum = gaussian_filter(result_sum, sigma=4)
    return result_sum

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--obj', type=str, default='<Object_name>')
parser.add_argument('--data_type', type=str, default='mvtec')
parser.add_argument('--dataset_path', type=str, default='./data/<CLASS_NAME>')
parser.add_argument('--reflection_path', type=str, default='./reflection/<CLASS_NAME>')
parser.add_argument('--output_path',type=str,default='./output')
parser.add_argument('--checkpoint_dir', type=str, default='.')
parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--img_resize', type=int, default=128)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

class_name=args.obj
output_path=args.output_path
if class_name in mip.CLASS_NAMES:
    args.dataset_path = './data/MIP'
    args.reflection_path = './reflection/MIP'
elif class_name in lego.CLASS_NAMES:
    args.dataset_path = './data/LEGO-3D'
    args.reflection_path = './reflection/LEGO-3D'
elif class_name in real.CLASS_NAMES:
    args.dataset_path = './data/Colmap'
    args.reflection_path = './reflection/Colmap'
dataset_path=args.dataset_path

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
fig_img_rocauc = ax[0]
fig_pixel_rocauc = ax[1]
img_size_1 = (400, 400)                       
x, y, mask = [], [], []
transform_mask = transforms.Compose([transforms.Resize(img_size_1, Image.NEAREST),
            transforms.ToTensor()]) 
img_dir = os.path.join(dataset_path, class_name, 'test')
gt_dir = os.path.join(
    dataset_path, class_name, 'ground_truth')
output_dir=os.path.join(output_path,class_name)
output_types=os.listdir(output_dir)
output_types.sort(key=lambda x:int(x.split('_')[-1]))
img_types = sorted(os.listdir(img_dir))
for img_type in img_types:
    img_type_dir = os.path.join(img_dir, img_type)
    if not os.path.isdir(img_type_dir):
        continue
    if class_name in real.CLASS_NAMES:
        img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                for f in os.listdir(img_type_dir)
                                if f.endswith('.jpg')])
    else:
        img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                for f in os.listdir(img_type_dir)
                                if f.endswith('.png')])
    x.extend(img_fpath_list)

    if img_type.split("_")[0] == 'good':
        y.extend([0] * len(img_fpath_list))
        mask.extend([None] * len(img_fpath_list))
    else:
        y.extend([1] * len(img_fpath_list))
        gt_type_dir = os.path.join(gt_dir, img_type)
        img_fname_list = [os.path.splitext(os.path.basename(f))[
            0] for f in img_fpath_list]
        if class_name in lego.CLASS_NAMES:
            gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                for img_fname in img_fname_list]
        else:
            gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                for img_fname in img_fname_list]
        mask.extend(gt_fpath_list)
test_imgs = list()
test_imgs_R = list()
gt_mask_list = list()
gt_list = y
score_map_list=list()
scores=list()
pred_list=list()
recon_imgs=list()
recon_imgs_R=list()
gt_list_array=np.array(gt_list)

for i in range(len(x)):
    if y[i] == 0:
        gt_mask = torch.zeros([1, img_size_1[0], img_size_1[1]])
    else:
        gt_mask = Image.open(mask[i])
        gt_mask = transform_mask(gt_mask)
    

    gt_mask_list.extend(gt_mask.cpu().numpy())


front=gt_list.index(0)
end=len(gt_list)
gt_array=np.arange(front,end)
gt_list_2=[1]*len(gt_list)

MSE_loss = nn.MSELoss(reduction='none')
tfms = transforms.Compose([
    transforms.Resize(img_size_1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
with torch.no_grad():
    for i,anomaly_type in enumerate(output_types):
        ref_path=os.path.join(output_dir,anomaly_type,'ref.png')
        rgb_path=os.path.join(output_dir,anomaly_type,'rgb8.png')
        Rref_path=os.path.join(output_dir,anomaly_type,'Rref.png')
        Rrgb_path=os.path.join(output_dir,anomaly_type,'Rrgb8.png')
        ref=tfms(Image.open(ref_path).convert('RGB')).unsqueeze(0).cuda()
        rgb=tfms(Image.open(rgb_path).convert('RGB')).unsqueeze(0).cuda()
        Rref=tfms(Image.open(Rref_path).convert('RGB')).unsqueeze(0).cuda()
        Rrgb=tfms(Image.open(Rrgb_path).convert('RGB')).unsqueeze(0).cuda()
        ref_feature=model(ref)
        rgb_feature=model(rgb)
        Rref_feature=model(Rref)
        Rrgb_feature=model(Rrgb)
        score = 0
        for i in range(len(ref_feature)):
            s_act1 = ref_feature[i]
            s_act2 = Rref_feature[i]
            mse_loss = MSE_loss(s_act1, rgb_feature[i]).sum(1, keepdim=True) * MSE_loss(s_act2, Rrgb_feature[i]).sum(1, keepdim=True)
            score += F.interpolate(mse_loss, size=img_size_1, mode='bilinear', align_corners=False)
        
        score = score.squeeze(1).cpu().numpy()
        for i in range(score.shape[0]):
            score[i] = gaussian_filter(score[i], sigma=4)
        recon_imgs.extend(rgb.cpu().numpy())
        recon_imgs_R.extend(Rrgb.cpu().numpy())
        test_imgs.extend(ref.cpu().numpy())
        test_imgs_R.extend(Rref.cpu().numpy())
        scores.append(score)

scores = np.asarray(scores).squeeze()
np.save("scores_padgs.npy",scores)
max_anomaly_score = scores.max() 
min_anomaly_score = scores.min() 
scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
gt_mask = np.asarray(gt_mask_list).round()
precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
a = 2 * precision * recall   
b = precision + recall
f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
threshold = thresholds[np.argmax(f1)]

fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
gt_list = np.asarray(gt_list)
img_roc_auc = roc_auc_score(gt_list, img_scores)
print('image ROCAUC: %.3f' % (img_roc_auc))

plt.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
plt.legend(loc="lower right")
print("threshold", threshold)
pre_view_save_dir = os.path.join("./AD_result", class_name)
os.makedirs(pre_view_save_dir,exist_ok=True)
plot_fig2(test_imgs, recon_imgs, scores, gt_mask_list, threshold, pre_view_save_dir,class_name)
print(class_name)