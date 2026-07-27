import argparse
import json
import re
import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from datasets.retinal_patch import RetinalFullImageDataset, RetinalPatchDataset
from tensorboardX import SummaryWriter
from models.model_factory import available_model_names, build_model
from models.vmunet.sobel_guidance import available_sobel_variants

from engine import *
import os
import sys
from datetime import datetime
from pathlib import Path

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def resolve_model_config(config):
    if config.network not in config.model_configs:
        raise ValueError(f'Unknown model variant: {config.network}')

    model_config = dict(config.model_configs[config.network])
    if config.initialization == 'scratch':
        model_config['load_ckpt_path'] = None
    elif config.initialization == 'vmamba':
        model_config['load_ckpt_path'] = config.vmamba_checkpoint_path
    else:
        raise ValueError(
            f'Unsupported initialization policy: {config.initialization}'
        )
    return model_config


def configure_sobel_ablation(config, args, parser):
    """Inject explicit Sobel settings without changing other model variants."""
    is_sobel_model = config.network == 'vmunet_highres_sobel'
    supplied_sobel_argument = (
        args.sobel_operator is not None
        or args.sobel_q is not None
        or args.sobel_selected_config is not None
    )
    if supplied_sobel_argument and not is_sobel_model:
        parser.error(
            'Sobel operator arguments require '
            '--model vmunet_highres_sobel'
        )
    if not is_sobel_model:
        return
    if args.sobel_selected_config is not None:
        if args.sobel_operator is not None or args.sobel_q is not None:
            parser.error(
                '--sobel-selected-config cannot be combined with explicit '
                '--sobel-operator/--sobel-q'
            )
        selected_path = Path(args.sobel_selected_config)
        if not selected_path.is_file():
            parser.error(
                'Selected Sobel configuration not found: {}'.format(
                    selected_path
                )
            )
        selected = json.loads(selected_path.read_text(encoding='utf-8'))
        try:
            args.sobel_operator = selected['operator']
            args.sobel_q = float(
                selected['q_by_dataset'][config.datasets]
            )
            args.guidance_strength = float(
                selected['guidance_strength']
            )
            config.seed = int(selected['seed'])
            config.initialization = selected['initialization']
        except (KeyError, TypeError, ValueError) as error:
            parser.error(
                'Invalid selected Sobel configuration {}: {}'.format(
                    selected_path,
                    error,
                )
            )
    if args.sobel_operator is None:
        parser.error('--sobel-operator is required for the Sobel model')
    if args.sobel_operator not in available_sobel_variants():
        parser.error(
            'Unknown selected Sobel operator: {}'.format(
                args.sobel_operator
            )
        )
    if args.sobel_q is None or args.sobel_q <= 0:
        parser.error('--sobel-q must be a positive training-set statistic')
    if args.guidance_strength < 0:
        parser.error('--guidance-strength must be non-negative')

    sobel_config = dict(config.model_configs['vmunet_highres_sobel'])
    sobel_config.update(
        sobel_operator=args.sobel_operator,
        sobel_q=args.sobel_q,
        guidance_strength=args.guidance_strength,
    )
    model_configs = dict(config.model_configs)
    model_configs['vmunet_highres_sobel'] = sobel_config
    config.model_configs = model_configs



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    if config.retinal_patch_mode:
        train_dataset = RetinalPatchDataset(
            os.path.join(config.data_path, 'train'),
            samples_per_epoch=config.samples_per_epoch,
            patch_size=config.patch_size,
            patch_stride=config.train_patch_stride,
            positive_crop_probability=config.positive_crop_probability,
            min_fov_fraction=config.min_fov_fraction,
            horizontal_flip_probability=config.horizontal_flip_probability,
            vertical_flip_probability=config.vertical_flip_probability,
            rotation_probability=config.rotation_probability,
            photometric_probability=config.photometric_probability,
            gamma_probability=config.gamma_probability,
            return_fov_mask=config.model_config.get(
                'uses_sobel_guidance', False
            ),
            fov_erosion_radius=config.model_config.get(
                'fov_erosion_radius', 2
            ),
        )
        val_dataset = RetinalFullImageDataset(os.path.join(config.data_path, 'val'))
        test_dataset = RetinalFullImageDataset(os.path.join(config.data_path, 'test'))
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
        )
    else:
        train_dataset = NPY_datasets(config.data_path, config, train=True)
        train_loader = DataLoader(train_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=config.num_workers)
        val_dataset = NPY_datasets(config.data_path, config, train=False)
        val_loader = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=config.num_workers,
                                    drop_last=True)
        test_loader = val_loader





    print('#----------Prepareing Model----------#')
    model_cfg = resolve_model_config(config)
    config.model_config = model_cfg
    logger.info(
        'Model variant: %s; initialization: %s; checkpoint: %s',
        config.network,
        config.initialization,
        model_cfg['load_ckpt_path'],
    )
    model = build_model(config.network, model_cfg)
    model.load_from()
    model = model.cuda()

    profile_size = config.patch_size if config.retinal_patch_mode else config.input_size_h
    cal_params_flops(model, profile_size, logger)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = None if config.fixed_learning_rate else get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    best_dice = -1.0
    start_epoch = 1
    min_epoch = 1

    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        if config.retinal_patch_mode:
            evaluate_retinal_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
                split_name='test',
                save_predictions=True,
            )
        else:
            test_one_epoch(
                    test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                )
        return




    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        best_dice = checkpoint.get('best_dice', best_dice)

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        if config.retinal_patch_mode:
            val_metrics = evaluate_retinal_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
                split_name='val',
                epoch=epoch,
            )
            loss = val_metrics['loss']
            if val_metrics['dice'] > best_dice:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                best_dice = val_metrics['dice']
                min_loss = loss
                min_epoch = epoch
        else:
            loss = val_one_epoch(
                    val_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config
                )
            if loss < min_loss:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'best_dice': best_dice,
                'model_name': config.network,
                'initialization': config.initialization,
                'seed': config.seed,
                'model_config': dict(config.model_config),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        if config.retinal_patch_mode:
            evaluate_retinal_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
                split_name='test',
                save_predictions=True,
            )
            best_name = f'best-epoch{min_epoch}-dice{best_dice:.4f}.pth'
        else:
            test_one_epoch(
                    test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                )
            best_name = f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth'
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, best_name)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('DRIVE', 'STARE'), default=None)
    parser.add_argument(
        '--model',
        choices=available_model_names(),
        default=None,
        help='Model variant. The default keeps the original VM-UNet baseline.',
    )
    parser.add_argument(
        '--initialization',
        choices=('scratch', 'vmamba'),
        default='scratch',
        help=(
            'Weight initialization policy. scratch is the default for '
            'architecture ablations; vmamba enables compatible pre-training.'
        ),
    )
    parser.add_argument(
        '--run-tag',
        default=None,
        help='Optional label added to the result directory, for example baseline.',
    )
    parser.add_argument(
        '--sobel-operator',
        choices=available_sobel_variants(),
        default=None,
        help='Fixed Sobel operator used only by vmunet_highres_sobel.',
    )
    parser.add_argument(
        '--sobel-q',
        type=float,
        default=None,
        help='Training-set safe-FOV 0.99 quantile for the selected operator.',
    )
    parser.add_argument(
        '--sobel-selected-config',
        default=None,
        help=(
            'Final operator configuration produced by select_operator.py. '
            'Use this for later module ablations, not operator screening.'
        ),
    )
    parser.add_argument(
        '--guidance-strength',
        type=float,
        default=1.0,
        help='Lambda in E_guided = E * (1 + lambda * G); fixed to 1 in screening.',
    )
    parser.add_argument(
        '--result-root',
        default='results',
        help='Root output directory; Sobel screening uses its own subdirectory.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed. Sobel operator screening is locked to 42.',
    )
    args = parser.parse_args()
    config = setting_config
    if args.model is not None:
        config.network = args.model
    config.initialization = args.initialization
    config.seed = args.seed
    if args.dataset is not None:
        config.datasets = args.dataset
        config.data_path = f'./data/retinal_576/{args.dataset}/'
    configure_sobel_ablation(config, args, parser)
    config.model_config = resolve_model_config(config)
    if args.dataset is not None:
        if args.run_tag is not None and not re.fullmatch(
            r'[A-Za-z0-9_-]+', args.run_tag
        ):
            parser.error('--run-tag may contain only letters, digits, _ and -')
        run_name_parts = [config.network, args.dataset]
        if args.run_tag:
            run_name_parts.append(args.run_tag)
        run_name_parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
        config.work_dir = (
            args.result_root.rstrip('/\\')
            + '/'
            + '_'.join(run_name_parts)
            + '/'
        )
    main(config)
