import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# Assuming Equirec2Perspec_vr and Perspec2Equirec_vr are in app.processors.external
from app.processors.external.Equirec2Perspec_vr import Equirectangular as E2P_Equirectangular
from app.processors.external.Perspec2Equirec_vr import Perspective as P2E_Perspective

TEMP_DIR_VR = ".vr_temp_processing" # Define a temporary directory for VR processing files


def _get_sobel_kernels(device):
    sobel_x_kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=torch.float32).reshape(1, 1, 3, 3)
    sobel_y_kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device, dtype=torch.float32).reshape(1, 1, 3, 3)
    return sobel_x_kernel, sobel_y_kernel

class EquirectangularConverter:
    def __init__(self, equirect_image_data_rgb_uint8: np.ndarray, device: torch.device):
        """
        Initializes with equirectangular image data.
        :param equirect_image_data_rgb_uint8: NumPy array (H, W, C) in RGB, uint8 format.
        :param device: PyTorch device to use.
        """
        os.makedirs(TEMP_DIR_VR, exist_ok=True)
        self.device = device
        # Convert NumPy HWC RGB to Torch CHW RGB tensor on GPU
        self.equirect_tensor_cxhxw_rgb_uint8 = torch.from_numpy(
            equirect_image_data_rgb_uint8
        ).permute(2,0,1).to(self.device)
        
        self.channels, self.height, self.width = self.equirect_tensor_cxhxw_rgb_uint8.shape
        self.e2p_instance = E2P_Equirectangular(self.equirect_tensor_cxhxw_rgb_uint8)

    def calculate_theta_phi_from_bbox(self, bbox_np: np.ndarray):
        x1, y1, x2, y2 = map(int, bbox_np)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        theta = (x_center / self.width - 0.5) * 360.0
        phi = -(y_center / self.height - 0.5) * 180.0 # Negative because image y is top-to-bottom

        return theta, phi

    def get_perspective_crop(self, FOV: float, THETA: float, PHI: float, height: int, width: int) -> torch.Tensor:
        """
        Returns a perspective crop as a Torch tensor (C, H, W) in RGB, uint8 format, on GPU.
        """
        # E2P_Equirectangular.GetPerspective now returns a Torch tensor (CHW, RGB, uint8)
        persp_torch_cxhxw_rgb_uint8 = self.e2p_instance.GetPerspective(FOV, THETA, PHI, height, width)
        return persp_torch_cxhxw_rgb_uint8


class PerspectiveConverter:
    def __init__(self, base_equirect_image_data_rgb_uint8: np.ndarray, device: torch.device):
        """
        Initializes with the base equirectangular image data (used for dimensions and as background).
        :param base_equirect_image_data_rgb_uint8: NumPy array (H, W, C) in RGB, uint8 format.
        :param device: PyTorch device to use.
        """
        os.makedirs(TEMP_DIR_VR, exist_ok=True)

        self.device = device
        # Convert NumPy HWC RGB to Torch CHW RGB tensor on GPU
        self.base_equirect_tensor_cxhxw_rgb_uint8 = torch.from_numpy(
            base_equirect_image_data_rgb_uint8
        ).permute(2,0,1).to(self.device)
        self.orig_channels, self.orig_height, self.orig_width = self.base_equirect_tensor_cxhxw_rgb_uint8.shape
        self.sobel_x_kernel, self.sobel_y_kernel = _get_sobel_kernels(self.device)

    def _apply_feathering(self, mask_torch: torch.Tensor) -> torch.Tensor:
        """ Applies feathering to a Torch mask.
        :param mask_torch: Torch tensor (1, H, W) or (H, W), boolean or float, on GPU.
        :return: Feathered mask as Torch tensor (1, H, W), float, on GPU.
        """
        mask_float_torch = mask_torch.float()
        if mask_float_torch.ndim == 2: # HW
            mask_float_torch = mask_float_torch.unsqueeze(0) # 1HW
        if mask_float_torch.ndim == 3 and mask_float_torch.shape[0] != 1 : # CHW but C != 1
             mask_float_torch = mask_float_torch[0:1,:,:] # Take first channel

        # Add batch dimension for conv2d: (1, 1, H, W)
        mask_batch_channel = mask_float_torch.unsqueeze(0)

        gradient_x = F.conv2d(mask_batch_channel, self.sobel_x_kernel, padding=1)
        gradient_y = F.conv2d(mask_batch_channel, self.sobel_y_kernel, padding=1)

        gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        max_grad = torch.max(gradient_magnitude)

        if max_grad < 1e-5: # Use a small epsilon to handle near-zero gradients
            return mask_float_torch # Return original 1HW float mask

        feathered_mask_batch = 1.0 - gradient_magnitude / max_grad

        return feathered_mask_batch.squeeze(0) # Return 1HW float mask


    def stitch_single_perspective(self,
                                  target_equirect_torch_cxhxw_rgb_uint8: torch.Tensor,
                                  processed_crop_torch_cxhxw_rgb_uint8: torch.Tensor,
                                  theta: float, phi: float, fov: float,
                                  is_left_eye: bool):
        """
        Stitches a single processed perspective crop back into the target equirectangular image.
        Modifies target_equirect_torch_cxhxw_rgb_uint8 in place.
        Assumes all tensors are on self.device.
        """

        p2e_instance = P2E_Perspective(processed_crop_torch_cxhxw_rgb_uint8, FOV=fov, THETA=theta, PHI=phi)
        # GetEquirec returns Torch tensors:
        # equirect_component_torch: (C, H, W) RGB uint8, the processed crop warped to equirectangular space.
        # mask_torch_original_shape: (1, H, W) boolean, indicating valid warped pixels.
        equirect_component_torch, mask_torch_original_shape = p2e_instance.GetEquirec(self.orig_height, self.orig_width)

        # Create eye region mask (1, H, W) to define the current eye's hemisphere
        eye_region_mask = torch.zeros_like(mask_torch_original_shape, dtype=torch.bool) 
        half_width = self.orig_width // 2
        if is_left_eye:
            eye_region_mask[:, :, :half_width] = True
        else:
            eye_region_mask[:, :, half_width:] = True

        # Apply eye region mask to the original projection mask to make it eye-specific
        eye_specific_mask_torch_original_shape = mask_torch_original_shape & eye_region_mask

        # Feather the eye-specific mask
        feathered_mask_torch_float_1hw = self._apply_feathering(eye_specific_mask_torch_original_shape) # Returns 1HW float

        target_equirect_float = target_equirect_torch_cxhxw_rgb_uint8.float() / 255.0

        # Use the original, non-eye-masked equirect_component_torch for pixel data
        equirect_component_float = equirect_component_torch.float() / 255.0

        # feathered_mask_torch_float_1hw is (1, H, W), can be broadcast with (C, H, W)
        composite_float = target_equirect_float * (1.0 - feathered_mask_torch_float_1hw) + \
                          equirect_component_float * feathered_mask_torch_float_1hw


        # Use the eye-specific (non-feathered) mask for direct replacement areas
        # eye_specific_mask_torch_original_shape is (1, H, W) boolean
        # Expand to (C, H, W) for torch.where
        mask_for_where = eye_specific_mask_torch_original_shape.expand_as(target_equirect_float)
        
        final_blended_float = torch.where(mask_for_where, composite_float, target_equirect_float)
        
        target_equirect_torch_cxhxw_rgb_uint8[:] = (torch.clamp(final_blended_float * 255.0, 0, 255)).byte()

        # Explicitly delete intermediate tensors if memory is tight, though Python's GC + PyTorch should handle it.
        del p2e_instance, equirect_component_torch, mask_torch_original_shape
        del eye_region_mask, eye_specific_mask_torch_original_shape 
        del feathered_mask_torch_float_1hw
        del target_equirect_float, equirect_component_float, composite_float, mask_for_where, final_blended_float
 
def cleanup_temp_dir():
    import shutil
    if os.path.exists(TEMP_DIR_VR):
        shutil.rmtree(TEMP_DIR_VR)