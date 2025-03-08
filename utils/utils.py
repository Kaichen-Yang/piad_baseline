from copy import deepcopy
import importlib
import json
import os
from operator import index

import cv2
import imageio
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity
from utils.model_helper import ModelHelper
from retrieval.loftr import LoFTR, default_cfg
from retrieval.retrieval import *
from easydict import EasyDict
from arguments import ModelParams, PipelineParams, OptimizationParams
import yaml
import ipdb


# efficientLoFTR
import sys
sys.path.append('./EfficientLoFTR')
from src.loftr import LoFTR as EfficientLoFTR
from src.loftr import default_cfg as Efficient_default_cfg
from src.config.default import get_cfg_defaults
from src.lightning.lightning_loftr import PL_LoFTR, reparameter

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='./data/MADreal/',
                        help='path to folder with synthetic or llff data')
    parser.add_argument('--config', is_config_file=True, default='configs/MADreal/Bear.txt',
                        help='config file path')
    parser.add_argument("--model_name", type=str,
                        help='name of the nerf model')
    parser.add_argument("--output_dir", type=str, default='./output/',
                        help='where to store output images/videos')
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts',
                        help='folder with saved checkpoints')
    parser.add_argument("--Rckpt_dir", type=str, default='./Rckpts',
                        help='folder with saved reflection checkpoints')
    parser.add_argument("--ckpt_name", type=str, 
                        help='name of ckpt')
    
    # training gaussian model
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--chunk", type=int, default=1024*32,  # 1024*32
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,  # 1024*64
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    # rendering options
    # parser.add_argument("--iteration", default=-1, type=int)

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--class_name", type=str, default='Bear',
                        help='LEGO-3D anomaly class')

    # llff options
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')

    # Pose opt options
    parser.add_argument("--lrate", type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument("--iter_num", type=int, default=300,
                        help='pose opt iteration')
    
    # parameters to define initial pose
    parser.add_argument("--delta_psi", type=float, default=0.0,
                        help='Rotate camera around x axis')
    parser.add_argument("--delta_phi", type=float, default=0.0,
                        help='Rotate camera around z axis')
    parser.add_argument("--delta_theta", type=float, default=0.0,
                        help='Rotate camera around y axis')
    parser.add_argument("--delta_t", type=float, default=0.0,
                        help='translation of camera (negative = zoom in)')
    
    # unfold reflection & illumination
    parser.add_argument('--reflection_dir', type=str, default="./reflection/OBJECT")
    # ratio are recommended to be 3-5, bigger ratio will lead to over-exposure 
    parser.add_argument('--ratio', type=int, default=5)
    # model path
    parser.add_argument('--Decom_model_low_path', type=str, default="./retrieval/model/init_low.pth")
    parser.add_argument('--unfolding_model_path', type=str, default="./retrieval/model/unfolding.pth")
    parser.add_argument('--adjust_model_path', type=str, default="../retrieval/model/L_adjust.pth")

    return parser, lp, op, pp


def rot_psi(phi): return np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]])


def rot_theta(th): return np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]])


def rot_phi(psi): return np.array([
    [np.cos(psi), -np.sin(psi), 0, 0],
    [np.sin(psi), np.cos(psi), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])


def trans_t(t): return np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]])


def load_blender(data_dir, model_name, obs_img_num, half_res, white_bkgd, *kwargs):

    with open(os.path.join(data_dir + str(model_name) + "/obs_imgs/", 'transforms.json'), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']

    img_path = os.path.join(data_dir + str(model_name) +
                            "/obs_imgs/", frames[obs_img_num]['file_path'] + '.png')
    img_rgba = imageio.imread(img_path)
    # rgba image of type float32
    img_rgba = (np.array(img_rgba) / 255.).astype(np.float32)
    H, W = img_rgba.shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    if white_bkgd:
        img_rgb = img_rgba[..., :3] * \
            img_rgba[..., -1:] + (1. - img_rgba[..., -1:])
    else:
        img_rgb = img_rgba[..., :3]
    imageio.imwrite("horse.png",img_rgb)
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    img_rgb = np.asarray(img_rgb*255, dtype=np.uint8)
    obs_img_pose = np.array(
        frames[obs_img_num]['transform_matrix']).astype(np.float32)
    phi, theta, psi, t = kwargs
    start_pose = trans_t(t) @ rot_phi(phi/180.*np.pi) @ rot_theta(theta /
                                                                  180.*np.pi) @ rot_psi(psi/180.*np.pi)  @ obs_img_pose
    # image of type uint8
    return img_rgb, [H, W, focal], start_pose, obs_img_pose


def load_blender_AD(data_dir, model_name, obs_img_num, half_res, white_bkgd, method,**kwargs):
    # load train nerf images and poses
    meta = {}
    with open(os.path.join(data_dir, str(model_name), 'transforms_train.json'), 'r') as fp:
        meta["train"] = json.load(fp)
        imgs = []
        poses = []
        for frame in meta["train"]['frames']:
            fname = os.path.join(data_dir, str(model_name), frame['file_path'])+'.png'
            img = imageio.imread(fname)
            img = (np.array(img) / 255.).astype(np.float32)
            if white_bkgd:
                img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
            else:
                img = img[..., :3]
            img = np.asarray(img*255, dtype=np.uint8)
            # img = tensorify(img)
            pose = (np.array(frame['transform_matrix']))
            pose = np.array(pose).astype(np.float32)

            imgs.append(img)
            poses.append(pose)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # imgs=np.array(imgs)
    imgs = np.stack(imgs, 0)
    poses = np.array(poses)

    H, W = int(H), int(W)

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    img_path = os.path.join(data_dir, str(model_name),'anomaly',str(obs_img_num)+".png")
    img_rgba = imageio.imread(img_path)
    # rgba image of type float32
    img_rgba = (np.array(img_rgba) / 255.).astype(np.float32)
    if white_bkgd and img_rgba.shape[-1]==4:
        img_rgb = img_rgba[..., :3] * \
            img_rgba[..., -1:] + (1. - img_rgba[..., -1:])
    else:
        img_rgb = img_rgba[..., :3]
    img_rgb = np.asarray(img_rgb*255, dtype=np.uint8)
    index_best = 0
    score_best = 0.5
    initial_pose = np.zeros([4, 4])
    
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i in range(len(imgs)):
            imgs_half_res[i] = cv2.resize(imgs[i], (W, H), interpolation=cv2.INTER_AREA)
        imgs=imgs_half_res
    
    # use lpips
    if method=='lpips':
        for i in range(len(imgs)):
            score=calculate_lpips(imgs[i],img_rgb,'vgg')
            if score < score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("lpips_min:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    elif method=='ssim':
    # use SSIM
        for i in range(len(imgs)):
            score,_=calculate_resmaps(imgs[i],img_rgb,'ssim')
            if score > score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("SSIM_max:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    # image of type uint8
    return img_rgb, [H, W, focal],initial_pose,score_best

def load_blender_ad(data_dir, model_name, white_bkgd):
    # load train nerf images and poses
    meta = {}
    with open(os.path.join(data_dir, str(model_name), 'transforms.json'), 'r') as fp:
        meta["train"] = json.load(fp)
        imgs = []
        poses = []
        for frame in meta["train"]['frames']:
            fname = os.path.join(data_dir, str(model_name), frame['file_path'])
            img = imageio.imread(fname)
            img = (np.array(img) / 255.).astype(np.float32)
            if white_bkgd and img.shape[-1]==4:
                img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
            else:
                img = img[..., :3]
            img = np.asarray(img*255, dtype=np.uint8)
            # img = tensorify(img)
            pose = (np.array(frame['transform_matrix']))
            pose = np.array(pose).astype(np.float32)
            imgs.append(img)
            poses.append(pose)
    
    imgs = np.stack(imgs, 0)
    poses = np.array(poses)
    

    imgs_half_res = np.zeros((imgs.shape[0], 400, 400, 3)) # np.zeros((imgs.shape[0], H, W, 3))
    for i in range(len(imgs)):
        imgs_half_res[i] = cv2.resize(imgs[i], (400, 400), interpolation=cv2.INTER_AREA)
    imgs=imgs_half_res.astype(np.uint8)
    return imgs, poses

def load_colmap_ad(scene, resize):
    meta = {}
    imgs = []
    poses = []
    for c in scene.train_cameras[1.0]:
        img = c.original_image
        invp = np.zeros((4,4))
        invp[:3,:3] = c.R.transpose(1,0)
        invp[:3,3] = c.T
        invp[3,3] = 1
        pose = np.linalg.inv(invp)
        pose[:3, 1:3] *= -1
        # tensor调整维度顺序


        img = cv2.resize(img.permute(1, 2, 0).cpu().numpy(), (resize, resize), interpolation=cv2.INTER_AREA)


        imgs.append(img)
        poses.append(pose)
            
        
    imgs = np.stack(imgs, 0)
    poses = np.array(poses)
        
    return imgs,poses

    """# To adapt madreal dataset ADD BY YKC
    imgs_half_res = np.zeros((imgs.shape[0], 400, 400, 3))
    for i in range(len(imgs)):
        imgs_half_res[i] = cv2.resize(imgs[i], (400, 400), interpolation=cv2.INTER_AREA)
    imgs=imgs_half_res.astype(np.uint8)
    
    
    poses = np.array(poses)
    
    return imgs,poses"""

def find_nearest(imgs,obs_img,poses,method):
    # use lpips
    score_best=0.5
    if method=='lpips':
        for i in range(len(imgs)):
            score=calculate_lpips(imgs[i],obs_img,'vgg')
            if score < score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("lpips_min:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    elif method=='ssim':
    # use SSIM
        for i in range(len(imgs)):
            score,_=calculate_resmaps(imgs[i],obs_img,'ssim')
            if score > score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
    elif method=='l2':
        for i in range(len(imgs)):
            score,_=calculate_resmaps(imgs[i],obs_img,'l2')
            if score > score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("SSIM_max:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    return initial_pose
        
def calculate_lpips(img1,img2,net='vgg',use_gpu=True):
    ## Initializing the model
    loss_fn = lpips.LPIPS(net)
    img1 = lpips.im2tensor(img1)  # RGB image from [-1,1]
    img2 = lpips.im2tensor(img2)

    if use_gpu:
        img1 = img1.cuda()
        img2 = img2.cuda()
    score = loss_fn.forward(img1, img2)
    return score

    
def resmaps_ssim(img_input,img_pred):
    score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
    return score,resmap

def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps

def resmaps_l1(imgs_input, imgs_pred):
    resmaps = np.abs(imgs_input - imgs_pred)
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps

def calculate_resmaps(img_input, img_pred, method, dtype="float64"):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # if RGB, transform to grayscale and reduce tensor dimension to 3
    if img_input.shape[-1] == 3:
        img_input_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_pred_gray = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
    else:
        img_input_gray = img_input
        img_pred_gray = img_pred

    # calculate remaps
    if method == "l2":
        scores, resmaps = resmaps_l2(img_input_gray, img_pred_gray)
    elif method in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(img_input_gray, img_pred_gray)
    # if dtype == "uint8":
        # resmaps = img_as_ubyte(resmaps)
    return scores, resmaps
def rgb2bgr(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr
def bgr2rgb(img_bgr):
    img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    return img_rgb

def show_img(title, img_rgb):  # img - rgb image
    img_bgr = rgb2bgr(img_rgb)
    cv2.imwrite(title, img_bgr)
    # cv2.imshow(title, img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def find_POI(img_rgb, DEBUG=False):  # img - RGB image in range 0...255
    """find some points to location.---"""
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    if DEBUG:
        img = cv2.drawKeypoints(img_gray, keypoints, img)
        show_img("Detected_points.png", img)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy  # pixel coordinates


# Misc
def img2mse(x, y): return torch.mean((x - y) ** 2)
# ADD BY YKC
def img2mae(x, y): return torch.mean(torch.abs(x - y))
# ADD BY YKC
def Lpose(x, y, e): return torch.mean(torch.mul(e, torch.abs(x - y)))

# Copy from SparseGS  https://formycat.github.io/SparseGS-Real-Time-360-Sparse-View-Synthesis-using-Gaussian-Splatting/
def pearson_depth_loss(depth_src, depth_target):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co

def depth_loss(depth_src, depth_target, mask):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target * mask).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co

def depth_loss2(depth_src, depth_target):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co

def mse(imageA, imageB):
    err = torch.mean((imageA - imageB) ** 2)
    return err

def MAPE(x,y):return torch.mean(torch.abs((x-y)/y))
def Relative_L2(x,y):return torch.mean(torch.abs((x-y)**2/y**2))
def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

# Load llff data

# Slightly modified version of LLFF data loading code
# see https://github.com/Fyusion/LLFF for original


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any(
        [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg,
                        '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(
            len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    def p34_to_44(p): return np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i,
                                [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(
        p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]

    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, bds


def load_llff_data(data_dir, model_name, obs_img_num, *kwargs, factor=8, recenter=True, bd_factor=.75, spherify=False):
    # factor=8 downsamples original imgs by 8x
    poses, bds, imgs = _load_data(
        data_dir + str(model_name) + "/", factor=factor)
    print('Loaded', data_dir + str(model_name) + "/", bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, bds = spherify_poses(poses, bds)

    #images = images.astype(np.float32)
    images = np.asarray(images * 255, dtype=np.uint8)
    poses = poses.astype(np.float32)
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    obs_img = images[obs_img_num]
    obs_img_pose = np.concatenate(
        (poses[obs_img_num], np.array([[0, 0, 0, 1.]])), axis=0)
    phi, theta, psi, t = kwargs
    start_pose = rot_phi(phi/180.*np.pi) @ rot_theta(theta/180. *
                                                     np.pi) @ rot_psi(psi/180.*np.pi) @ trans_t(t) @ obs_img_pose
    return obs_img, hwf, start_pose, obs_img_pose, bds


def pose_retrieval(imgs,obs_img,poses):
    # Prepare model.
    model = load_model(pretrained_model='./retrieval/model/net_best.pth', use_gpu=True)

    # Extract database features.
    gallery_feature = extract_feature(model=model, imgs=imgs)

    # Query.
    query_image = transform_query_image(obs_img)

    # Extract query features.
    query_feature = extract_feature_query(model=model, img=query_image)
    
    # Sort.
    similarity, index = sort_img(query_feature, gallery_feature)

    return poses[index[0]]


def pose_retrieval_efficient(imgs,obs_img,poses):
    # Prepare model.
    model = load_model_efficient()

    # Extract database features.
    gallery_feature = extract_feature_efficient(model=model, imgs=imgs)

    # Extract query features.
    query_feature = extract_feature_query_efficient(model=model, img=obs_img)
    # ipdb.set_trace()
    # Sort.
    similarity, index = sort_img_efficient(query_feature, gallery_feature)

    return poses[index]

def pose_retrieval_loftr(imgs,obs_img,poses):
    """find the most similar pose from poses, return poses[q]. """
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("retrieval/model/indoor_ds_new.ckpt")['state_dict'])
    # num_params = count_parameters(matcher)#11561456
    # ipdb.set_trace()
    matcher = matcher.eval().cuda()
    if obs_img.shape[-1] == 3:
        query_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY)
    img0 = torch.from_numpy(query_img)[None][None].cuda() / 255.
    max_match=-1
    max_index=-1
    for i in range(len(imgs)):
        if imgs[i].shape[-1] == 3:
            gallery_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        img1 = torch.from_numpy(gallery_img)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            # conf = batch['conf_matrix']
            # match_num = (conf > 0.2).sum()
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
        match_num=len(mconf)
        if match_num>max_match:
            max_match=match_num
            max_index=i
    return poses[max_index]

def pose_retrieval_efficientloftr(imgs,obs_img,poses):
    dist_list = [np.sum(np.abs(img-obs_img)) for img in imgs]
    # 取出排名前50%的图片
    top_num = int(len(imgs)*0.5) +1
    top_index = np.argsort(dist_list)[:top_num]
    # 取出top_num的图片
    imgs = imgs[top_index]
    poses = poses[top_index]
    
    _default_cfg = deepcopy(Efficient_default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    _default_cfg['coarse']['npe'] = [832, 832, 832, 832]
    matcher = EfficientLoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load('./EfficientLoFTR/weights/eloftr_outdoor.ckpt')['state_dict'])
    matcher = reparameter(matcher)
    # num_params = count_parameters(matcher)
    # ipdb.set_trace()
    matcher = matcher.eval().cuda()
    if obs_img.shape[-1] == 3:
        query_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY)
    img0 = torch.from_numpy(query_img)[None][None].cuda() / 255. 
    max_match=-1
    max_index=-1
    for i in range(len(imgs)):
        if imgs[i].shape[-1] == 3:
            gallery_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        img1 = torch.from_numpy(gallery_img)[None][None].cuda()    # origin: .cuda()/255.
        if (img1>1.1).sum():                                         # ADD BY YKC
            img1=img1/255.
        batch = {'image0': img0, 'image1': img1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            conf = batch['conf_matrix']
            match_num = (conf > 0.7).sum()                  # origin==0.2 BY YKC  # TODO
            # mkpts0 = batch['mkpts0_f'].cpu().numpy()
            # mkpts1 = batch['mkpts1_f'].cpu().numpy()
            # mconf = batch['mconf'].cpu().numpy()
        # match_num=len(mconf)
        # print(match_num)
        if match_num>max_match:
            max_match=match_num
            max_index=i
    return poses[max_index]

def pose_retrieval_efficientloftr_cross(imgs,Rimgs,obs_img,Robs_img,poses):
    dist_list = [np.sum(np.abs(img-obs_img)) for img in imgs]
    # 取出排名前50%的图片
    top_num = int(len(imgs)*0.5) +1
    top_index = np.argsort(dist_list)[:top_num]
    # 取出top_num的图片
    imgs = imgs[top_index]
    Rimgs = Rimgs[top_index]
    poses = poses[top_index]
    
    _default_cfg = deepcopy(Efficient_default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    _default_cfg['coarse']['npe'] = [832, 832, 832, 832]
    matcher = EfficientLoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load('./EfficientLoFTR/weights/eloftr_outdoor.ckpt')['state_dict'])
    matcher = reparameter(matcher)
    # num_params = count_parameters(matcher)
    # ipdb.set_trace()
    matcher = matcher.eval().cuda()
    if obs_img.shape[-1] == 3:
        query_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY)
    if Robs_img.shape[-1] == 3:
        Rquery_img = cv2.cvtColor(Robs_img, cv2.COLOR_RGB2GRAY)
    img0 = torch.from_numpy(query_img)[None][None].cuda() / 255. 
    Rimg0 = torch.from_numpy(Rquery_img)[None][None].cuda() / 255.
    max_match=-1
    max_index=-1
    for i in range(len(imgs)):
        if imgs[i].shape[-1] == 3:
            gallery_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        if Rimgs[i].shape[-1] == 3:
            Rgallery_img = cv2.cvtColor(Rimgs[i], cv2.COLOR_RGB2GRAY)
        img1 = torch.from_numpy(gallery_img)[None][None].cuda()    # origin: .cuda()/255.
        Rimg1 = torch.from_numpy(Rgallery_img)[None][None].cuda()
        if (img1>1.1).sum():                                         # ADD BY YKC
            img1=img1/255.
        if (Rimg1>1.1).sum():                                         
            Rimg1=Rimg1/255.
        batch_N = {'image0': img0, 'image1': img1}
        batch_R = {'image0': Rimg0, 'image1': Rimg1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch_N)
            Nconf = batch_N['conf_matrix']
            match_num = (Nconf > 0.7).sum()                  # origin==0.2 BY YKC  # TODO
            matcher(batch_R)
            Rconf = batch_R['conf_matrix']
            match_num += (Rconf > 0.7).sum()*2
            # mkpts0 = batch['mkpts0_f'].cpu().numpy()
            # mkpts1 = batch['mkpts1_f'].cpu().numpy()
            # mconf = batch['mconf'].cpu().numpy()
        # match_num=len(mconf)
        # print(match_num)
        if match_num>max_match:
            max_match=match_num
            max_index=i
    return poses[max_index]

def pose_retrieval_efficientloftr2(imgs, obs_img, poses):
    matcher = PL_LoFTR(config=get_cfg_defaults(), pretrained_ckpt='/data/baizeyu/EfficientLoFTR/weights/eloftr_outdoor.ckpt')
    if obs_img.shape[-1] == 3:
        query_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY)
    img0 = torch.from_numpy(query_img)[None][None].cuda() / 255.
    max_match=-1
    max_index=-1
    for i in range(len(imgs)):
        if imgs[i].shape[-1] == 3:
            gallery_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        img1 = torch.from_numpy(gallery_img)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}
    ret_dict = matcher.test_step(batch)
    match_num = ret_dict['num_matches']
    if match_num>max_match:
            max_match=match_num
            max_index=i
    return poses[max_index]

def pose_retrieval_efficientloftr4(imgs, obs_img, poses,thr):
    # imgs: (200, 400, 400, 3)
    # obs_img: (400, 400, 3)
    # 计算imgs到obs_img的l1距离，得到距离列表
    dist_list = [np.sum(np.abs(img-obs_img)) for img in imgs]
    # 取出排名前50%的图片
    top_num = int(len(imgs)*0.5)
    top_index = np.argsort(dist_list)[:top_num]
    # 取出top_num的图片
    imgs = imgs[top_index]
    poses = poses[top_index]

    model_type = 'full' # 'full' for best quality, 'opt' for best efficiency

    # You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
    precision = 'fp32' # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

    # You can also change the default values like thr. and npe (based on input image size)

    if model_type == 'full':
        _default_cfg = deepcopy(full_default_cfg)
    elif model_type == 'opt':
        _default_cfg = deepcopy(opt_default_cfg)
        
    if precision == 'mp':
        _default_cfg['mp'] = True
    elif precision == 'fp16':
        _default_cfg['half'] = True
    _default_cfg['match_coarse']['thr'] = thr
    matcher = EfficientLoFTR(config=_default_cfg)

    matcher.load_state_dict(torch.load("../EfficientLoFTR/weights/eloftr_outdoor.ckpt")['state_dict'])
    matcher = reparameter(matcher) # no reparameterization will lead to low performance

    if precision == 'fp16':
        matcher = matcher.half()

    matcher = matcher.eval().cuda()


    if obs_img.shape[-1] == 3:
        query_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY)
    img0 = torch.from_numpy(query_img)[None][None].cuda() / 255.
    
    max_match=-1
    max_index=-1
    for i in range(len(imgs)):
        if imgs[i].shape[-1] == 3:
            gallery_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        img1 = torch.from_numpy(gallery_img)[None][None].cuda()
        if (img1>1.1).sum():
            img1=img1/255.
        batch = {'image0': img0, 'image1': img1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            conf = batch['conf_matrix']
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            # match_num = (conf > 0.2).sum()
            mconf = batch['mconf'].cpu().numpy()
        match_num=len(mconf)
        # match_num=len(mkpts0)
        if match_num>max_match:
            max_match=match_num
            max_index=i
            _batch = batch
        # 绘图
    mkpts0 = _batch['mkpts0_f'].cpu().numpy()
    mkpts1 = _batch['mkpts1_f'].cpu().numpy()
    mconf = _batch['mconf'].cpu().numpy()
    if model_type == 'opt':
        print(mconf.max())
        mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))

    color = cm.jet(mconf)
    text = [
            'EfficientLoFTR',
            'Matches: {}'.format(len(mkpts0)),
    ]
    fig = make_matching_figure(obs_img, imgs[max_index], mkpts0, mkpts1, color, text=text)
    return poses[max_index], fig, max_index

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)