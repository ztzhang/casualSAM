import os
import logging
import configargparse
import json
from slam.slamopt import DepthVideoOptimization
from slam.depth_video_new_cache import get_video_dataset
import time
import pprint
from util.util_3dvideo import depth_to_points_Rt



def parse_args(input_string=None):
    parser = configargparse.ArgParser()
    parser.add_argument('--config', type=str, is_config_file=True, default='./experiments_slam/davis/default_config.yaml',
                        help='config file path')
    parser.add_argument('--gpu', type=str, default='1', help='gpu id')
    parser.add_argument('--track_name', type=str,
                        help='track name')
    parser.add_argument('--dataset_name', type=str,
                        help='dataset name')
    parser.add_argument('--log_dir', type=str,
                        help='log dir')
    parser.add_argument('--prefix', type=str, default='', help='prefix')
    parser.add_argument('--buffer_size', type=int, default=5,
                        help='buffer size for moving window')
    parser.add_argument('--cache_length', type=int, default=300,
                        help='cache length for moving window')
    parser.add_argument('--has_gt', action='store_true',
                        help='whether to use ground truth')
    parser.add_argument('--loss_fn', type=str, default='mse',
                        help='type of loss function')
    parser.add_argument('--optimization_layers',
                        action="append", help='optimization layers')
    parser.add_argument('--frame_list', type=int, action="append",
                        default=None, help='optimization layers')
    parser.add_argument('--num_workers', type=int,
                        default=1, help='number of workers')
    parser.add_argument('--tensorboard_dir', type=str,
                        default='', help='tensorboard dir')
    parser.add_argument('--translation_lr', type=float,
                        default=1e-3, help='output path')
    parser.add_argument('--rotation_lr', type=float,
                        default=1e-3, help='output path')
    parser.add_argument('--scale_lr', type=float,
                        default=1e-4, help='output path')
    parser.add_argument('--depth_lr', type=float,
                        default=1e-4, help='output path')
    parser.add_argument('--uncertainty_lr', type=float,
                        default=1e-2, help='uncertainty branch learning rate')
    parser.add_argument('--intrinsics_lr', type=float,
                        default=1e-5, help='lr for intrinsics')
    parser.add_argument('--scale_iter', type=int,
                        default=601, help='iteration for scale opt')
    parser.add_argument('--joint_iter', type=int,
                        default=801, help='iteration for scale opt')
    parser.add_argument('--use_inverse_scale',
                        action='store_true', help='scale = 1/(scale**2)')
    parser.add_argument('--depth_scale_init', type=float,
                        default=1e-2, help='initial scale for depth')
    parser.add_argument('--full_BA_iter', type=int,
                        default=1001, help='iterations for full BA opt')
    parser.add_argument('--print_to_file', action='store_true',
                        help='print log to file')
    parser.add_argument('--i_print_full_BA', type=int,
                        default=100, help='print loss every i steps.')
    parser.add_argument('--use_uncertainty',
                        action='store_true', help='use uncertainty')
    parser.add_argument('--not_load_mask', action='store_true',
                        help='do not load motion mask')
    parser.add_argument('--batch_size', type=int, default=40,
                        help='batch size for full BA')
    parser.add_argument('--max_width', type=int, default=1024,
                        help='max width for storing hi res images')
    parser.add_argument('--max_width_opt', type=int, default=512,
                        help='max width for optimizing over depth and pose')
    parser.add_argument('--opt_intrinsics', action='store_true',
                        help='optimize intrinsics as well')
    parser.add_argument('--i_print_init_opt', type=int, default=100,
                        help='print loss value per iteration during init')
    parser.add_argument('--frame_cap', type=int, default=None,
                        help='cap at max frame number')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='skip every k frames.')
    parser.add_argument('--init_focal_length', type=float,
                        default=0.55, help='initial focal length')
    parser.add_argument('--depth_regularizer', type=str,
                        default='none', help='regularizer for depth')
    parser.add_argument('--depth_reg_weight', type=float,
                        default=0.0, help='weight for depth reg')
    parser.add_argument('--depth_consistency_weight', type=float,
                        default=0.1, help='weight for depth consistency')
    parser.add_argument('--depth_net', type=str,
                        default='midas', help='decide optimization style')
    parser.add_argument('--image_sequence_stride', type=int, default=1,
                        help='stride for image sequence, i.e. sample every k frame in the sequence')
    parser.add_argument('--uncertainty_consistency_weight', type=float,
                        default=0, help='weight for uncertainty consistency')
    parser.add_argument('--scale_grid_size', type=int,
                        default=1, help='grid size for scale')
    parser.add_argument('--cauchy_bias', type=float, default=1,
                        help='bias term for the cauchy loss so that it is stable')
    parser.add_argument('--self_calibration', action='store_true')
    parser.add_argument('--use_pixel_weight_for_depth_reg',
                        action='store_true', help='use pixel weight for depth reg')
    parser.add_argument('--pixel_weight_scale', type=float,
                        default=1, help='sclae for pixel weight')
    parser.add_argument('--pixel_weight_bias', type=float,
                        default=1, help='sclae for pixel weight')
    parser.add_argument('--pixel_weight_normalization',
                        action='store_true', help='normalize pixel weight')
    parser.add_argument('--cap_uncertainty',
                        action='store_true', help='maxout uncertainty')
    parser.add_argument('--uncertainty_channels', type=int,
                        default=1, help='use K uncertainty channels')
    parser.add_argument(
        '--detach_uncertainty_from_depth_consistency', action='store_true', help='detach uncertainty from depth consistency')
    parser.add_argument('--use_ego_flow_for_disp_consistency',
                        action='store_true', help='use ego flow for disp consistency')
    parser.add_argument('--use_pose_init_reg',
                        action='store_true', help='use pose init reg')
    parser.add_argument('--pose_reg_weight', type=float,
                        default=1, help='weight for pose reg')
    parser.add_argument('--depth_only_BA',
                        action='store_true', help='use depth only BA')
    parser.add_argument('--scale_uncertainty', default=1.0,
                        type=float, help='scale uncertainty for depth only BA')
    parser.add_argument('--depth_reg_bias', default=0.01,
                        type=float, help='bias for depth reg')
    parser.add_argument('--add_uncertainty_only_BA',
                        action='store_true', help='add uncertainty only BA')
    parser.add_argument('--add_scale_only_BA',
                        action='store_true', help='add scale only BA')
    parser.add_argument('--new_init_frozen',
                        action='store_true', help='use new init')
    parser.add_argument('--load_init',
                        action='store_true', help='load init')
    parser.add_argument('--use_constant_uncertainty',
                        action='store_true', help='use constant uncertainty')
    parser.add_argument('--init_only', action='store_true', help='init only')
    parser.add_argument('--load_checkpoint_path', type=str,
                        default=None, help='checkpoint path')
    parser.add_argument('--uncertainty_only_iter', type=int,
                        default=101, help='iteration for uncertainty only BA')
    if input_string is not None:
        opt = parser.parse_args(input_string)
    else:
        opt = parser.parse_args()
    opt.date = time.strftime('%m-%d', time.localtime())
    return opt


def format_output_path(opt):
    optimization_layers_string = '_'.join(opt.optimization_layers)
    opt.optimization_layers_string = optimization_layers_string
    opt.output_path = os.path.join(opt.log_dir, opt.prefix.format(**vars(opt)))
    if opt.load_checkpoint_path is not None:
        opt.load_checkpoint_path = os.path.join(
            opt.log_dir, opt.load_checkpoint_path.format(**vars(opt)))
    opt.tensorboard_dir = os.path.join(
        opt.output_path, opt.prefix.format(**vars(opt)))


def train_fn_from_opt(opt):
    import torch
    format_output_path(opt)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(opt))
    print('output_path is: '+opt.output_path)
    gpu = opt.gpu
    os.makedirs(opt.output_path, exist_ok=True)
    if opt.print_to_file:

        logging.basicConfig(filename=os.path.join(opt.output_path, 'training_info.log'), level=logging.INFO,
                            filemode='w', format='%(asctime)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO,
                            filemode='w', format='%(asctime)s - %(message)s', datefmt='%m-%d %H:%M:%S')

    # device = torch.device(
    #    'cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    device = 'cuda:0'
    dataset = get_video_dataset(opt)
    dataset.to(device, cache_data=True)
    dataset.initialize_depth_scale()
    dataset.init_optimizer()
    dataset.initialize_depth_scale()
    dataset.store_initial_depth()
    dataset.store_initial_uncertainty_feat()
    logging.info('initialized dataset')
    logging.info('saving optimization hyperparameters')
    arg_dict = vars(opt)
    json_string = json.dumps(arg_dict, indent=4)
    os.makedirs(opt.output_path, exist_ok=True)
    with open(os.path.join(opt.output_path, 'opt.json'), 'w') as f:
        f.write(json_string)
    depth_opt = DepthVideoOptimization(opt, dataset)
    if opt.frame_list is None:
        frame_list = list(range(depth_opt.depth_video.number_of_frames))
    else:
        frame_list = opt.frame_list
    if opt.frame_cap is not None:
        frame_list = frame_list[:opt.frame_cap]
    frame_list = frame_list[::opt.frame_skip]

    if opt.depth_only_BA:
        K = dataset.camera_intrinsics.get_K_and_inv()[0].detach().cpu().numpy()
        logging.info('performing depth only BA')
        if opt.load_checkpoint_path is not None:
            logging.info(f'loading checkpoint from {opt.load_checkpoint_path}')
            dataset.load_checkpoint(os.path.join(
                opt.load_checkpoint_path, 'BA_full'))
        else:
            logging.info(f'loading checkpoint from {opt.output_path}')
            dataset.load_checkpoint(os.path.join(opt.output_path, 'BA_full'))
        dataset.reload_depth_net()
        dataset.reset_optimizer()
        depth_opt.BA_over_all_frames_low_mem(
            frame_list, iteration=opt.full_BA_iter//3, batch_size=opt.batch_size, use_pose_init_reg=opt.use_pose_init_reg, depth_only=True, scale_uncertainty=opt.scale_uncertainty)
        K = dataset.camera_intrinsics.get_K_and_inv()[0].detach().cpu().numpy()
        logging.info(K)
        depth_opt.save_results(os.path.join(
            opt.output_path, 'depth_only_refine'), frame_list=frame_list, with_uncertainty=True)
        return

    if opt.use_uncertainty:
        K = dataset.camera_intrinsics.get_K_and_inv()[0].detach().cpu().numpy()
        logging.info(K)
        if opt.load_init:
            dataset.load_checkpoint(os.path.join(
                opt.output_path, 'init_window_BA'))
            logging.info('loaded init')
        else:
            logging.info('running init')
            depth_opt.local_BA_init_uncertainty(frame_list, scale_only=True)
            depth_opt.save_results(os.path.join(
                opt.output_path, 'init_window_BA'), frame_list=frame_list, with_uncertainty=True)
        if opt.init_only:
            return
        K = dataset.camera_intrinsics.get_K_and_inv()[0].detach().cpu().numpy()
        logging.info(K)
        dataset.store_relative_pose()
        dataset.clear_flow_cache()
        depth_opt.depth_video.reset_optimizer()
        if opt.add_uncertainty_only_BA:
            logging.info('performing uncertainty only BA')
            depth_opt.BA_over_all_frames_low_mem(
                frame_list, iteration=opt.uncertainty_only_iter, batch_size=opt.batch_size, max_range=8, use_pose_init_reg=opt.use_pose_init_reg, uncertainty_only=True)
            K = dataset.camera_intrinsics.get_K_and_inv()[
                0].detach().cpu().numpy()
            logging.info(K)
            depth_opt.save_results(os.path.join(
                opt.output_path, 'uc_only_BA_init'), frame_list=frame_list, with_uncertainty=True)
        if opt.add_scale_only_BA:
            logging.info('performing scale only BA')
            depth_opt.BA_over_all_frames_low_mem(
                frame_list, iteration=300, batch_size=opt.batch_size, max_range=8, use_pose_init_reg=opt.use_pose_init_reg, scale_only=True)
            K = dataset.camera_intrinsics.get_K_and_inv()[
                0].detach().cpu().numpy()
            logging.info(K)
            depth_opt.save_results(os.path.join(
                opt.output_path, 'scale_only_BA_init'), frame_list=frame_list, with_uncertainty=True)

        depth_opt.BA_over_all_frames_low_mem(
            frame_list, iteration=opt.full_BA_iter//3, batch_size=opt.batch_size, max_range=16, use_pose_init_reg=opt.use_pose_init_reg)
        K = dataset.camera_intrinsics.get_K_and_inv()[0].detach().cpu().numpy()
        logging.info(K)
        depth_opt.save_results(os.path.join(
            opt.output_path, 'BA_4'), frame_list=frame_list, with_uncertainty=True)
        depth_opt.BA_over_all_frames_low_mem(
            frame_list, iteration=opt.full_BA_iter//3, batch_size=opt.batch_size, max_range=64, use_pose_init_reg=opt.use_pose_init_reg)
        K = dataset.camera_intrinsics.get_K_and_inv()[0].detach().cpu().numpy()
        logging.info(K)
        depth_opt.save_results(os.path.join(
            opt.output_path, 'BA_16'), frame_list=frame_list, with_uncertainty=True)
        depth_opt.BA_over_all_frames_low_mem(
            frame_list, iteration=opt.full_BA_iter//3, batch_size=opt.batch_size, use_pose_init_reg=opt.use_pose_init_reg)
        K = dataset.camera_intrinsics.get_K_and_inv()[0].detach().cpu().numpy()
        logging.info(K)
        depth_opt.save_results(os.path.join(
            opt.output_path, 'BA_full'), frame_list=frame_list, with_uncertainty=True)
        # dataset.reload_depth_net()
        # depth_opt.BA_over_all_frames_low_mem(
        #     frame_list, iteration=opt.full_BA_iter//3, batch_size=opt.batch_size, depth_only=True, scale_uncertainty=2)
        # depth_opt.save_results(os.path.join(
        #     opt.output_path, 'BA_depth_refine'), frame_list=frame_list, with_uncertainty=True)
    else:
        depth_opt.local_BA_init_full(frame_list, paired=True)
        depth_opt.save_results(os.path.join(
            opt.output_path, 'init_window_BA'), frame_list=frame_list)
        depth_opt.BA_over_all_frames_low_mem(
            frame_list, iteration=opt.full_BA_iter, batch_size=opt.batch_size)
        depth_opt.save_results(opt.output_path, frame_list=frame_list)
    print(f'{opt.track_name} is done. output is saved at {opt.output_path}')


def main():
    opt = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    train_fn_from_opt(opt)


if __name__ == '__main__':
    main()


'''
    depth_opt.init_pair(opt.frame_list[0], opt.frame_list[1])
    depth_opt.keyframe_buffer.buffer = frame_list
    # profiling loop

    all_pairs = list(itertools.combinations(
        depth_opt.keyframe_buffer.buffer, 2))

    with torch.no_grad():
        depths = depth_opt.depth_video.predict_depth(
            depth_opt.keyframe_buffer.buffer)

    depth_opt.active_optimizers = [
        depth_opt.depth_video.pose_optimizer, depth_opt.depth_video.depth_optimizer]

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(skip_first=10,
                                         wait=10, warmup=2, active=10, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            './log/scale_only_opt'),
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        for iter in range(101):
            losses = []
            for o in depth_opt.all_optimizers:
                o.zero_grad()
            loss = 0
            for i, j in all_pairs:
                idi = depth_opt.keyframe_buffer.buffer.index(i)
                idj = depth_opt.keyframe_buffer.buffer.index(j)
                pred = depth_opt.forwrard_prediction(
                    i, j, depths=[depths[idi:idi+1, ...], depths[idj:idj+1, ...]])
                losses.append(depth_opt.calculate_loss(pred)/len(all_pairs))
            # optimize all relative parameters
            total_loss = sum(losses)
            total_loss.backward()
            for o in depth_opt.active_optimizers:
                o.step()
            prof.step()
            # if iter % 10 == 0:
            print(iter)

'''
