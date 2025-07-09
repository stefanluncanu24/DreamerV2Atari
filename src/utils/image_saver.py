import torch
import torchvision.utils as vutils
import os

def save_reconstruction_predictions(recon_pred_tensor, output_dir, global_step, batch_idx):
    """
    Saves a batch of reconstructed prediction tensors as image files.

    Args:
        recon_pred_tensor (torch.Tensor): The tensor containing reconstructed images.
                                         Expected shape: (N, C, H, W) where N is batch size.
                                         Pixel values are expected to be in [-0.5, 0.5].
        output_dir (str): The directory where the images will be saved.
        global_step (int): The current global training step or frame number.
        batch_idx (int): The index of the current batch.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert from [-0.5, 0.5] to [0, 1] for saving
    recon_pred_tensor = recon_pred_tensor + 0.5

    # Ensure the tensor is on CPU before saving
    recon_pred_tensor = recon_pred_tensor.cpu()

    # Save each image in the batch
    for i in range(recon_pred_tensor.shape[0]):
        filename = os.path.join(output_dir, f"recon_pred_step{global_step}_batch{batch_idx}_img{i}.png")
        vutils.save_image(recon_pred_tensor[i], filename)
    print(f"Saved {recon_pred_tensor.shape[0]} reconstruction images to {output_dir}")
