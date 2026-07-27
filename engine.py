import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix, roc_auc_score
from PIL import Image
from pathlib import Path
from utils import save_imgs


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _supports_sobel_guidance(model):
    return bool(
        getattr(_unwrap_model(model), "supports_sobel_guidance", False)
    )


def _forward_model(
    model,
    images,
    safe_fov_mask=None,
    guidance_map=None,
):
    if not _supports_sobel_guidance(model):
        return model(images)
    return model(
        images,
        safe_fov_mask=safe_fov_mask,
        guidance_map=guidance_map,
    )


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += 1
        optimizer.zero_grad()
        safe_fov_mask = None
        if isinstance(data, dict):
            images = data["image"]
            targets = data["mask"]
            safe_fov_mask = data.get("safe_fov_mask")
        else:
            images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if safe_fov_mask is not None:
            safe_fov_mask = safe_fov_mask.cuda(non_blocking=True).float()

        out = _forward_model(
            model,
            images,
            safe_fov_mask=safe_fov_mask,
        )
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    if scheduler is not None:
        scheduler.step()
    return step


def _sliding_positions(image_size, patch_size, stride):
    positions = list(range(0, image_size - patch_size + 1, stride))
    last_position = image_size - patch_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def sliding_window_predict(
    image,
    model,
    patch_size=192,
    stride=96,
    batch_size=32,
    fov_mask=None,
):
    """Predict one normalized CHW image and blend overlapping patch probabilities."""
    _, height, width = image.shape
    y_positions = _sliding_positions(height, patch_size, stride)
    x_positions = _sliding_positions(width, patch_size, stride)
    coordinates = [(y, x) for y in y_positions for x in x_positions]
    probability_sum = torch.zeros((1, height, width), device=image.device)
    count_map = torch.zeros_like(probability_sum)

    full_guidance = None
    full_safe_fov = None
    if _supports_sobel_guidance(model):
        if fov_mask is None:
            raise ValueError("Sobel-guided sliding-window inference needs an FOV mask")
        model_core = _unwrap_model(model)
        full_guidance, full_safe_fov = model_core.prepare_full_guidance(
            image.unsqueeze(0),
            fov_mask.unsqueeze(0),
        )
        full_guidance = full_guidance[0]
        full_safe_fov = full_safe_fov[0]

    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        patches = torch.stack(
            [
                image[:, y : y + patch_size, x : x + patch_size]
                for y, x in batch_coordinates
            ]
        )
        guidance_patches = None
        safe_fov_patches = None
        if full_guidance is not None:
            guidance_patches = torch.stack(
                [
                    full_guidance[:, y : y + patch_size, x : x + patch_size]
                    for y, x in batch_coordinates
                ]
            )
            safe_fov_patches = torch.stack(
                [
                    full_safe_fov[:, y : y + patch_size, x : x + patch_size]
                    for y, x in batch_coordinates
                ]
            )
        predictions = _forward_model(
            model,
            patches,
            safe_fov_mask=safe_fov_patches,
            guidance_map=guidance_patches,
        )
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        for prediction, (y, x) in zip(predictions, batch_coordinates):
            probability_sum[:, y : y + patch_size, x : x + patch_size] += prediction
            count_map[:, y : y + patch_size, x : x + patch_size] += 1

    return (probability_sum / count_map.clamp_min(1)).unsqueeze(0)


def _binary_metrics(probabilities, targets, threshold, evaluation_masks=None):
    probabilities = np.concatenate(probabilities).reshape(-1)
    targets = np.concatenate(targets).reshape(-1).astype(np.uint8)
    if evaluation_masks is not None:
        evaluation_mask = np.concatenate(evaluation_masks).reshape(-1).astype(bool)
        probabilities = probabilities[evaluation_mask]
        targets = targets[evaluation_mask]
    predictions = (probabilities >= threshold).astype(np.uint8)
    confusion = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    miou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    auc = roc_auc_score(targets, probabilities) if np.unique(targets).size == 2 else float("nan")
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "dice": dice,
        "miou": miou,
        "auc": auc,
        "confusion_matrix": confusion,
    }


def evaluate_retinal_epoch(
    data_loader,
    model,
    criterion,
    logger,
    config,
    split_name,
    epoch=None,
    save_predictions=False,
):
    """Sliding-window full-image evaluation for retinal patch training."""
    model.eval()
    losses = []
    probabilities = []
    targets = []
    fov_masks = []
    output_dir = Path(config.work_dir) / "outputs" / split_name
    if save_predictions:
        output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for data in tqdm(data_loader):
            image = data["image"].cuda(non_blocking=True).float()
            mask = data["mask"].cuda(non_blocking=True).float()
            fov_mask = data["fov_mask"].cuda(non_blocking=True).float()
            case_name = data["case_name"][0]
            full_probability = sliding_window_predict(
                image=image[0],
                model=model,
                patch_size=config.patch_size,
                stride=config.inference_patch_stride,
                batch_size=config.inference_batch_size,
                fov_mask=fov_mask[0],
            )
            loss = criterion(full_probability, mask)
            losses.append(loss.item())
            probabilities.append(full_probability.squeeze().cpu().numpy()[None])
            targets.append(mask.squeeze().cpu().numpy()[None])
            fov_masks.append(fov_mask.squeeze().cpu().numpy()[None])

            if save_predictions:
                prediction = (
                    full_probability.squeeze().cpu().numpy() >= config.threshold
                ).astype(np.uint8)
                prediction *= fov_mask.squeeze().cpu().numpy().astype(np.uint8)
                Image.fromarray(prediction * 255).save(
                    output_dir / f"{case_name}_pred.png"
                )

    metrics = _binary_metrics(
        probabilities, targets, config.threshold, evaluation_masks=fov_masks
    )
    whole_image_metrics = _binary_metrics(
        probabilities, targets, config.threshold
    )
    metrics["loss"] = float(np.mean(losses))
    metrics["whole_image"] = whole_image_metrics
    prefix = split_name if epoch is None else f"{split_name} epoch: {epoch}"
    log_info = (
        f"{prefix}, loss (whole image): {metrics['loss']:.4f}; "
        f"FOV-only miou: {metrics['miou']:.4f}, "
        f"dice: {metrics['dice']:.4f}, accuracy: {metrics['accuracy']:.4f}, "
        f"specificity: {metrics['specificity']:.4f}, "
        f"sensitivity: {metrics['sensitivity']:.4f}, auc: {metrics['auc']:.4f}, "
        f"confusion_matrix: {metrics['confusion_matrix']}; "
        f"whole-image miou: {whole_image_metrics['miou']:.4f}, "
        f"dice: {whole_image_metrics['dice']:.4f}, "
        f"accuracy: {whole_image_metrics['accuracy']:.4f}, "
        f"specificity: {whole_image_metrics['specificity']:.4f}, "
        f"sensitivity: {whole_image_metrics['sensitivity']:.4f}, "
        f"auc: {whole_image_metrics['auc']:.4f}"
    )
    print(log_info)
    logger.info(log_info)
    return metrics


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
