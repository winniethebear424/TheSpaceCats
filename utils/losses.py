# Builds the masked loss function used during training.
# Loss is computed ONLY on the masked patches so the model
# is forced to infer missing regions, not memorize inputs.


def masked_loss(recon, target, mask, loss_type="smooth_l1", ssim_w=0.5):
    """
    Computes the masked loss between the reconstructed and target images.
    Only the patches where the mask is True are considered in the loss computation.

    Args:
        recon:     (B, C, H, W) model output
        target:    (B, C, H, W) ground-truth pixels
        mask:      (B, H, W) bool — True = was masked
        loss_type: one of l1 | l2 | smooth_l1 | ssim | combo
        ssim_w:    SSIM weight when loss_type == 'combo'

    Returns:
        scalar loss over masked pixels only
    """
    m = mask.unsqueeze(1).float()  # (B, 1, H, W)
    r, t = recon * m, target * m

    if loss_type == "l1":
        return F.l1_loss(r, t)
    elif loss_type == "l2":
        return F.mse_loss(r, t)
    elif loss_type == "smooth_l1":
        return F.smooth_l1_loss(r, t)
    elif loss_type == "ssim":
        return 1 - ssim(r, t, data_range=1.0)
    elif loss_type == "combo":
        l = F.smooth_l1_loss(r, t)
        s = 1 - ssim(r, t, data_range=1.0)
        return (1 - ssim_w) * l + ssim_w * s
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")