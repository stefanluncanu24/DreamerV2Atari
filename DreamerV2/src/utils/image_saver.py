import torch
import torchvision.utils as vutils
import os

def save_reconstruction_predictions(recon_pred_tensor, output_dir, global_step, batch_idx):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert back
    recon_pred_tensor = recon_pred_tensor + 0.5

    recon_pred_tensor = recon_pred_tensor.cpu()

    for i in range(recon_pred_tensor.shape[0]):
        filename = os.path.join(output_dir, f"recon_pred_step{global_step}_batch{batch_idx}_img{i}.png")
        vutils.save_image(recon_pred_tensor[i], filename)
    print(f"Saved {recon_pred_tensor.shape[0]} reconstruction images to {output_dir}")
