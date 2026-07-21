import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DSMEnhancedRegularizationLoss(nn.Module):
    def __init__(self, sigma_h=0.1, kernel_size=3, lambda_reg=0.1, alpha=0.05):
        super(DSMEnhancedRegularizationLoss, self).__init__()
        self.sigma_h = sigma_h
        self.kernel_size = kernel_size
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.padding = (kernel_size - 1) // 2

    def compute_dsm_weights(self, dsm):
        if dsm.dim() == 2:
            dsm = dsm.unsqueeze(0).unsqueeze(0)
        dsm_patches = F.unfold(dsm, kernel_size=self.kernel_size, padding=self.padding)
        dsm_center = dsm_patches[:, (self.kernel_size**2) // 2, :].unsqueeze(1)
        dsm_diff = dsm_patches - dsm_center
        h_sum = dsm_patches + dsm_center + 1e-6
        weights = torch.exp(- (dsm_diff ** 2) / (self.sigma_h**2 * h_sum**2))
        weights[:, (self.kernel_size**2)//2, :] = 0
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-6
        weights = weights / weights_sum
        return weights.squeeze(0).permute(1, 0)

    def compute_entropy(self, abundance):
        eps = 1e-6
        entropy = - (abundance * (abundance + eps).log()).sum(dim=1, keepdim=True)
        return entropy  # (1,1,H,W)

    def compute_dsm_gradient(self, dsm):
        if dsm.dim() == 2:
            dsm = dsm.unsqueeze(0).unsqueeze(0)
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=dsm.dtype, device=dsm.device).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=dsm.dtype, device=dsm.device).unsqueeze(0) / 8.0
        grad_x = F.conv2d(dsm, sobel_x, padding=1)
        grad_y = F.conv2d(dsm, sobel_y, padding=1)
        return torch.cat([grad_x, grad_y], dim=1)  # (1,2,H,W)

    def forward(self, abundance, dsm):
        assert abundance.dim() == 4, "Shape must be (1, M, H, W)"
        _, M, H, W = abundance.shape
        A_flat = abundance.view(M, -1)  # (M, H*W)

        weights = self.compute_dsm_weights(dsm)  # (H*W, k^2)

        A_patches = F.unfold(abundance, kernel_size=self.kernel_size, padding=self.padding)  # (1, M*k^2, H*W)
        A_patches = A_patches.view(M, self.kernel_size**2, -1).permute(2, 0, 1)  # (H*W, M, k^2)
        A_center = A_flat.permute(1, 0).unsqueeze(-1)  # (H*W, M, 1)

        grad = self.compute_dsm_gradient(dsm)  # (1,2,H,W)
        grad_unfold = F.unfold(grad, kernel_size=self.kernel_size, padding=self.padding)  # (1, 2*k^2, H*W)
        grad_unfold = grad_unfold.view(2, self.kernel_size**2, -1).permute(2, 0, 1)  # (H*W, 2, k^2)
        grad_center = grad.view(2, -1).permute(1, 0).unsqueeze(-1)  # (H*W, 2, 1)
        grad_diff = grad_unfold - grad_center  # (H*W, 2, k^2)

        grad_diff_exp = grad_diff.unsqueeze(1).repeat(1, M, 1, 1)  # (H*W, M, 2, k^2)
        diff_exp = (A_patches - A_center).unsqueeze(2)  # (H*W, M, 1, k^2)
        direction_diff = diff_exp - self.alpha * grad_diff_exp  # (H*W, M, 2, k^2)
        diff_norm = torch.norm(direction_diff, p=2, dim=2)  # (H*W, M, k^2)

        entropy = self.compute_entropy(abundance).view(-1)  # (H*W,)
        H_i = entropy.unsqueeze(1)  # (H*W, 1)
        H_j = entropy.unsqueeze(1).expand(-1, self.kernel_size ** 2)  # (H*W, k^2)
        entropy_term = 1 + H_i + H_j  # (H*W, k^2)
        entropy_term = entropy_term.unsqueeze(1).expand(-1, M, -1)  # (H*W, M, k^2)

        valid_mask = (weights > 0).float()
        weighted_diff = entropy_term * weights.unsqueeze(1) * diff_norm**2 * valid_mask.unsqueeze(1)  # (H*W, M, k^2)

        spatial_loss = weighted_diff.sum() / (valid_mask.sum() * M + 1e-6)

        return self.lambda_reg * spatial_loss


def SAD(y_true, y_pred):


    y_true_norm = torch.nn.functional.normalize(y_true, dim=1)
    y_pred_norm = torch.nn.functional.normalize(y_pred, dim=1)


    cos_similarity = torch.sum(y_true_norm * y_pred_norm, dim=1)
    cos_similarity = torch.clamp(cos_similarity, -1.0, 1.0)


    sad = torch.acos(cos_similarity)


    return torch.mean(sad)

def RMSE(y_true, y_pred):
    return torch.sqrt(torch.nn.functional.mse_loss(y_true, y_pred))

class NonZeroClipper(object): 

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)
