import os
import random
import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
import shutil
import datasets.MIP as mip
import datasets.LEGO_3D as lego
import datasets.MIPreal as real
from datasets.MIP import MIPDataset
from datasets.LEGO_3D import LEGODataset
from datasets.MIPreal import MIPrealDataset
from utils.camera_transf import camera_transf
from utils.utils import to8b
from utils.utils import (config_parser, img2mae, load_blender_ad, load_colmap_ad, pose_retrieval_efficientloftr)
from scene import Scene
from scene.cameras import Camera
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_pose, render_gaussians
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import GaussianModel
import math
import uuid
from utils.image_utils import psnr
from argparse import Namespace
from random import randint
from Retinex.RL_separate import Separate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 1024
random.seed(seed)
torch.manual_seed(seed)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def pose2camera(pose, obs_img, FovX):
    fovx = FovX
    posecopy = pose.clone()
    posecopy[:3, 1:3] *= -1
    w2c = np.linalg.inv(posecopy.cpu().detach().numpy())
    R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    image = torch.from_numpy(obs_img)  
    loaded_mask = None
    fovy = focal2fov(fov2focal(fovx, image.shape[2]), image.shape[1])
    FovY = fovy
    FovX = fovx
    camera = Camera(R=R, T=T,
                    FoVx=FovX, FoVy=FovY,
                    image=image, gt_alpha_mask=loaded_mask)

    return camera

def pose2viewmatrix(pose):
    posecopy = pose.clone()
    posecopy[:3, 1:3] *= -1
    w2c = torch.inverse(posecopy)
    Rt = torch.zeros([4, 4])
    Rt[:3, :3] = w2c[:3, :3]
    Rt[:3, 3] = w2c[:3, 3]
    Rt[3, 3] = 1.0

    
    return Rt.cuda()

def vec2ss_matrix(vector):

    ss_matrix = torch.zeros((3, 3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

def retinex(args, data_type):

    dataset_dir = args.source_path
    output_dir = args.reflection_path
    if os.path.exists(output_dir+"/test/good") or os.path.exists(output_dir+"/test/good_1"):
        print("Use Reflection images in '",output_dir, "'.")
        return
    
    # load model
    model = Separate(args).cuda()
    # make_dir
    if data_type != "Colmap":
        train_in_dir = os.path.join(dataset_dir,"train")
        train_out_dir = os.path.join(output_dir,"train")
        test_in_dir = os.path.join(dataset_dir,"test")
        test_out_dir = os.path.join(output_dir,"test")
        os.makedirs(train_out_dir, exist_ok=True)
        shutil.copy(os.path.join(dataset_dir,"transforms.json"), os.path.join(output_dir,"transforms.json"))
        shutil.copytree(os.path.join(dataset_dir,"ground_truth"), os.path.join(output_dir,"ground_truth"))
    else:
        train_in_dir = os.path.join(dataset_dir,"images")
        train_out_dir = os.path.join(output_dir,"images")
        test_in_dir = os.path.join(dataset_dir,"test")
        test_out_dir = os.path.join(output_dir,"test")
        os.makedirs(train_out_dir, exist_ok=True)
        shutil.copytree(os.path.join(dataset_dir,"sparse"), os.path.join(output_dir,"sparse"))
        shutil.copytree(os.path.join(dataset_dir,"ground_truth"), os.path.join(output_dir,"ground_truth"))
    print("================================= Separate image to Reflection & illumination =============================")
    if data_type == "MIP" or data_type == "Colmap":
        for filename in tqdm(os.listdir(train_in_dir),"Train"):
            raw_img_path = os.path.join(train_in_dir,filename)
            out_img_path = os.path.join(train_out_dir,filename)
            model.run(raw_img_path,out_img_path)
    else:
        os.makedirs(train_out_dir + '/good', exist_ok=True)
        for filename in tqdm(os.listdir(train_in_dir + '/good'),"Train"):
            raw_img_path = os.path.join(train_in_dir + '/good',filename)
            out_img_path = os.path.join(train_out_dir + '/good',filename)
            model.run(raw_img_path,out_img_path)
    for filename in os.listdir(test_in_dir):
        type_in_dir = os.path.join(test_in_dir,filename)
        type_out_dir = os.path.join(test_out_dir,filename)
        os.makedirs(type_out_dir, exist_ok=True)
        for img in tqdm(os.listdir(type_in_dir),f"Test:{filename}"):
            raw_img_path = os.path.join(type_in_dir,img)
            out_img_path = os.path.join(type_out_dir,img)
            model.run(raw_img_path,out_img_path)
    print("================================= Separate Finish ============================")




def build_gaussian_model(dataset, opt, pipe, 
                         testing_iterations, 
                         saving_iterations, 
                         checkpoint_iterations, 
                         checkpoint, 
                         debug_from, 
                         train_dir,
                         out_model_dir,
                         is_R=False):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, train_dir, out_model_dir)
    
    if scene.loaded_iter:
        bg_color = scene.getTrainCameras()[0].original_image[:,5,5]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return gaussians, scene, background

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    bg_color = scene.getTrainCameras()[0].original_image[:,5,5]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    white_background = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if (iteration - 1) <= 3000:
            # In the first stage, we use white bkgd to avoid num of points == 0.
            render_pkg = render_gaussians(viewpoint_cam, gaussians, pipe, white_background)
        else:
            # In the second stage, we use correct bkgd_color to prune big gaussians in background.
            render_pkg = render_gaussians(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_gaussians, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    if is_R:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.004, scene.cameras_extent, size_threshold)
                    else:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, train_dir, out_model_dir, shuffle=False)
    return gaussians, scene, background

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output ckpt folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    print("Output Rckpt folder: {}".format(args.Rmodel_path))
    os.makedirs(args.Rmodel_path, exist_ok = True)
    with open(os.path.join(args.Rmodel_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

        torch.cuda.empty_cache()

def run():
    # Parameters
    parser, lp, op, pp = config_parser()
    args = parser.parse_args()
    args.model_path = os.path.join(str(args.ckpt_dir), str(args.ckpt_name))
    args.source_path = os.path.join(str(args.data_dir), str(args.model_name))
    args.Rmodel_path = os.path.join(str(args.Rckpt_dir), str(args.ckpt_name))
    args.reflection_path = os.path.join(str(args.reflection_dir), str(args.model_name))
    data_type = str(args.data_dir).split("/")[-1]
    print(args.model_path)
    output_dir = args.output_dir
    model_name = args.model_name
    lrate = args.lrate
    iter_num = args.iter_num
    beta = 0.6

    
    with torch.no_grad():

        # Use Retinex separate images
        retinex(args, data_type)
        # Load pretrained 3DGS model
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    print("======================================== Build Reflection Gaussian ======================================")
    gaussians_R, scene_R, background_R = build_gaussian_model(lp.extract(args), 
                                                            op.extract(args), 
                                                            pp.extract(args), 
                                                            args.test_iterations, 
                                                            args.save_iterations, 
                                                            args.checkpoint_iterations, 
                                                            args.start_checkpoint, 
                                                            args.debug_from,
                                                            args.reflection_path,
                                                            args.Rmodel_path,
                                                            True)
    print("========================================== Build Normal Gaussian ========================================")
    gaussians, scene, background = build_gaussian_model(lp.extract(args), 
                                                            op.extract(args), 
                                                            pp.extract(args), 
                                                            args.test_iterations, 
                                                            args.save_iterations, 
                                                            args.checkpoint_iterations, 
                                                            args.start_checkpoint, 
                                                            args.debug_from,
                                                            args.source_path,
                                                            args.model_path)
        

    print("======= Load AD Dataset =======")
    if data_type == "MIP":
        class_names = mip.CLASS_NAMES if args.class_name == 'all' else [args.class_name]
    elif data_type == "LEGO-3D":
        class_names = lego.CLASS_NAMES if args.class_name == 'all' else [args.class_name]
    elif data_type == "Colmap":
        class_names = real.CLASS_NAMES if args.class_name == 'all' else [args.class_name]
        iter_num = 500
    else:
        print("This dataset neither MIP nor MAD-Sim nor Colmap, we can't read it. But you can add it at here.")

    for class_name in class_names:
        # load the good imgs with their poses
        if data_type == "Colmap":
            imgs, poses = load_colmap_ad(scene_R, 400)
        else:
            imgs, poses = load_blender_ad(
                args.reflection_dir, model_name, args.white_background)

        FovX = scene_R.train_cameras[1.0][0].FoVx

        # load the anomaly image
        if data_type == "MIP":
            dataset_info = MIPDataset(args=args,
                                    class_name=class_name,
                                    resize=400)
        elif data_type == "LEGO-3D":
            dataset_info = LEGODataset(args=args,
                                    class_name=class_name,
                                    resize=400)
        elif data_type == "Colmap":
            dataset_info = MIPrealDataset(args=args,
                                    class_name=class_name,
                                    resize=400)
        else:
            print("This dataset neither MIP nor MAD-Sim, we can't read it. But you can add it at here.")

        data_loader = DataLoader(dataset=dataset_info,
                                batch_size=1,
                                pin_memory=False)

        test_imgs = list()
        gt_mask_list = list()
        gt_list = list()

        index = 0
        
        print("============================================ Pose Estimation ============================================")
        for img_R, y, mask, img in tqdm(data_loader):

            img_R = torch.squeeze(img_R)
            img = torch.squeeze(img)

            query_img_R = img_R.cpu().numpy()
            img = img.transpose(1, 2).transpose(0, 1)
            query_img = img.cpu().numpy()

            # Find the start pose by looking for the most similar image
            start_pose = pose_retrieval_efficientloftr(imgs, query_img_R, poses)

            img_R = img_R.transpose(1, 2).transpose(0, 1)
            test_imgs.extend(img_R.cpu().numpy())
            gt_list.extend(y.cpu().numpy())
            mask = (mask.cpu().numpy() / 255.0).astype(np.uint8)
            gt_mask_list.extend(mask)
            query_img_R = img_R.cpu().numpy()
            query_img_R = (np.array(query_img_R) / 255.).astype(np.float32)
            query_img = (np.array(query_img) / 255.).astype(np.float32)

            # Create pose transformation model
            start_pose = torch.Tensor(start_pose).to(device)

            cam_transf = camera_transf().to(device)
            optimizer = torch.optim.Adam(
                params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999))
            
            testsavedir = os.path.join(
                output_dir, model_name, str(model_name) + "_" + str(index))
            os.makedirs(testsavedir, exist_ok=True)

            if OVERLAY is True:
                gif_imgs = []

            loss_o = torch.tensor(0.0).to(device)
            
            for k in range(iter_num):

                # Pose matrix transformation (a small network)
                pose = cam_transf(start_pose)
                # inverse
                viewmatrix = pose2viewmatrix(pose)
                try:
                    pose_r = pose2camera(pose, query_img_R, FovX)

                except(np.linalg.LinAlgError, torch._C._LinAlgError):
                    continue
                
                optimizer.zero_grad()

                render_img_R = render_pose(pose_r, gaussians_R, args, background_R, viewmatrix)["render"]
                render_img = render_pose(pose_r, gaussians, args, background, viewmatrix)["render"]
                query_img_R_tensor = torch.tensor(query_img_R)
                query_img_tensor = torch.tensor(query_img)
                loss = beta * img2mae(render_img_R, query_img_R_tensor) + (1 - beta) * img2mae(render_img, query_img_tensor)

                loss.backward(retain_graph=True)
                
                optimizer.step()

                new_lrate = lrate * (0.8 ** ((k + 1) / 100))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate

                if (k + 1) % 50 == 0 or k == 0:
                    #print('Step: ', k)
                    #print('Loss: ', loss)

                    if OVERLAY is True:
                        with torch.no_grad():
                            pose_r = pose2camera(pose, query_img_R, FovX)
                            rgb = render_pose(pose_r, gaussians_R, args, background_R, viewmatrix)["render"]
                            

                            # blend image
                            rgb = rgb.cpu().detach().numpy().astype(np.float32)
                            rgb8 = to8b(rgb)
                            rgb8_t = np.transpose(rgb8, (1, 2, 0))
                            ref = to8b(query_img_R)
                            ref_t = np.transpose(ref, (1, 2, 0))
                            filename = os.path.join(testsavedir, str(k) + '.png')
                            dst = cv2.addWeighted(rgb8_t, 0.7, ref_t, 0.3, 0)
                            imageio.imwrite(filename, dst)
                            gif_imgs.append(dst)

                            """
                            # rendered depth image
                            dep = dep - depmin
                            dep = dep * render_depth_mask
                            dep = (dep - 0)/(dep.max() - 0)
                            dep = dep.expand(3,400,400)
                            #dep[dep < 0] = 2
                            dep = dep.cpu().detach().numpy().astype(np.float32)
                            dep8 = to8b(dep)
                            dep8_t = np.transpose(dep8, (1, 2, 0))
                            imageio.imwrite(os.path.join(testsavedir, str(k) + 'dep.png'), dep8_t)

                            # GT depth image
                            gtdep_show = gtdep_img_t.unsqueeze(2).expand(400,400,3)
                            gtdep_show = gtdep_show.cpu().detach().numpy().astype(np.float32)
                            # gtdep_show = gtdep_show/255
                            gtdep_show = to8b(gtdep_show)
                            imageio.imwrite(os.path.join(testsavedir, str(k) + 'gtdep.png'), gtdep_show)

                            # depth mask
                            render_depth_mask_show = gt_depth_mask.expand(3,400,400)
                            render_depth_mask_show = render_depth_mask_show.cpu().detach().numpy().astype(np.float32)
                            render_depth_mask_show = to8b(render_depth_mask_show)
                            render_depth_mask_show_T = np.transpose(render_depth_mask_show, (1, 2, 0))
                            imageio.imwrite(os.path.join(testsavedir, str(k) + 'depmask.png'), render_depth_mask_show_T)
                            """
                            """
                            # grad
                            grad_img = render_img.grad.sum(dim=0).cpu().detach().numpy().astype(np.float32)
                            grad_max = grad_img.max()
                            grad_min = grad_img.min()
                            #grad_img = (grad_img - grad_min)/(grad_max - grad_min)
                            norm = matplotlib.colors.Normalize(vmin=0, vmax=grad_max)
                            plt.figure(figsize=(8, 8))
                            plt.imshow(grad_img, cmap='hot', norm=norm)
                            plt.colorbar()  
                            plt.title('Grad Heatmap')
                            plt.savefig(filename2)
                            #grad_img8 = to8b(grad_img)"""

                    if abs(loss-loss_o) < 3e-6:
                        #print('Break!!!')
                        #print('Step: ', k)
                        #print('Loss: ',loss)
                        break
                    

                    loss_o = loss.clone()

            Nrgb = render_pose(pose_r, gaussians, args, background, viewmatrix)["render"]
            Nrgb = Nrgb.cpu().detach().numpy().astype(np.float32)
            Nrgb8 = to8b(Nrgb)
            Nrgb8_t = np.transpose(Nrgb8, (1, 2, 0))
            Nref = to8b(query_img)
            Nref_t = np.transpose(Nref, (1, 2, 0))

            imageio.mimwrite(os.path.join(
                testsavedir, 'video.gif'), gif_imgs, fps=8)
            imageio.imwrite(os.path.join(testsavedir, 'Rref.png'), ref_t)
            imageio.imwrite(os.path.join(testsavedir, 'Rrgb8.png'), rgb8_t)
            imageio.imwrite(os.path.join(testsavedir, 'ref.png'), Nref_t)
            imageio.imwrite(os.path.join(testsavedir, 'rgb8.png'), Nrgb8_t)


            index = index + 1

DEBUG = False
OVERLAY = True

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    run()
