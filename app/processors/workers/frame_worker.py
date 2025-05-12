import traceback
from typing import TYPE_CHECKING
import threading
from math import floor, ceil
import os

import torch
from skimage import transform as trans
from torchvision.transforms import v2
import torchvision
from torchvision import transforms

import numpy as np

from app.processors.utils import faceutil
import app.ui.widgets.actions.common_actions as common_widget_actions # Used in original
from app.ui.widgets.actions import video_control_actions # Used in original
from app.helpers.miscellaneous import t512,t384,t256,t128, ParametersDict
from app.helpers.vr_utils import EquirectangularConverter, PerspectiveConverter # For VR180

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

torchvision.disable_beta_transforms_warning()

class FrameWorker(threading.Thread):
    def __init__(self, frame, main_window: 'MainWindow', frame_number, frame_queue, is_single_frame=False):
        super().__init__()
        self.frame_queue = frame_queue
        self.frame = frame # Expected to be HxWxC RGB uint8 NumPy array
        self.main_window = main_window
        self.frame_number = frame_number
        self.models_processor = main_window.models_processor
        self.video_processor = main_window.video_processor
        self.is_single_frame = is_single_frame
        self.parameters = {} # Will be populated from main_window.parameters
        # self.target_faces = main_window.target_faces # Not directly used here, main_window.target_faces is used
        # self.compare_images = [] # Not used in this merged version's flow
        self.is_view_face_compare: bool = False
        self.is_view_face_mask: bool = False

    def run(self):
        try:
            # Update parameters from markers (if exists)
            with self.main_window.models_processor.model_lock: # Ensure thread safety for UI data access
                video_control_actions.update_parameters_and_control_from_marker(self.main_window, self.frame_number)
            
            # It's safer to copy parameters that might be modified or accessed by UI thread
            self.parameters = self.main_window.parameters.copy() 
            current_control_state = self.main_window.control.copy()

            self.is_view_face_compare = self.main_window.faceCompareCheckBox.isChecked()
            self.is_view_face_mask = self.main_window.faceMaskCheckBox.isChecked()

            # Determine if any processing is needed
            needs_processing = (
                self.main_window.swapfacesButton.isChecked() or
                self.main_window.editFacesButton.isChecked() or
                current_control_state.get('FrameEnhancerEnableToggle', False) or
                current_control_state.get('VR180ModeEnableToggle', False) # VR180 always processes
            )

            if needs_processing:
                if not self.frame.flags['C_CONTIGUOUS']: # Ensure input frame is C-contiguous
                    self.frame = np.ascontiguousarray(self.frame)
                # process_frame returns BGR, uint8
                processed_frame_bgr_np_uint8 = self.process_frame(current_control_state)
                # Ensure output is C-contiguous for Qt display
                self.frame = np.ascontiguousarray(processed_frame_bgr_np_uint8)
            else:
                # If no processing, just convert RGB to BGR for display
                self.frame = self.frame[..., ::-1]
                self.frame = np.ascontiguousarray(self.frame)

            pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, self.frame)

            if self.video_processor.file_type == 'webcam' and not self.is_single_frame:
                self.video_processor.webcam_frame_processed_signal.emit(pixmap, self.frame)
            elif not self.is_single_frame:
                self.video_processor.frame_processed_signal.emit(self.frame_number, pixmap, self.frame)
            else: # Single frame processing (image or paused video)
                self.video_processor.single_frame_processed_signal.emit(self.frame_number, pixmap, self.frame)

            self.video_processor.frame_queue.get()
            self.video_processor.frame_queue.task_done()

            if self.video_processor.frame_queue.empty() and \
               not self.video_processor.processing and \
               self.video_processor.next_frame_to_display >= self.video_processor.max_frame_number:
                self.video_processor.stop_processing()

        except Exception as e:
            print(f"Error in FrameWorker for frame {self.frame_number}: {e}")
            traceback.print_exc()

    def _process_single_vr_perspective_crop(self,
                                 perspective_crop_torch_rgb_uint8: torch.Tensor,
                                 target_face_button: 'widget_components.TargetFaceCardButton',
                                 parameters_for_face: dict,
                                 control_global: dict,
                                 eye_side_for_debug: str = ""
                                 ) -> torch.Tensor:
        """
        Processes a single perspective crop for VR180: detects face, swaps, restores, pastes back.
        """
        # Detect face within the perspective crop
        # Note: input_size should match the crop's dimensions
        _, crop_kpss_5, _ = self.models_processor.run_detect(
            perspective_crop_torch_rgb_uint8,
            control_global['DetectorModelSelection'],
            max_num=1, # Assuming one primary face per targeted crop
            score=control_global['DetectorScoreSlider'] / 100.0,
            input_size=(perspective_crop_torch_rgb_uint8.shape[2], perspective_crop_torch_rgb_uint8.shape[1]), # W, H
            use_landmark_detection=control_global['LandmarkDetectToggle'],
            landmark_detect_mode=control_global['LandmarkDetectModelSelection'],
            landmark_score=control_global["LandmarkDetectScoreSlider"]/100.0,
            from_points=control_global["DetectFromPointsToggle"],
            rotation_angles=[0] # No auto-rotation for perspective crops
        )

        processed_crop_torch_rgb_uint8 = perspective_crop_torch_rgb_uint8.clone()

        if len(crop_kpss_5) > 0:
            face_kps_5_on_crop = crop_kpss_5[0] # Keypoints relative to the perspective crop

            # Get source embedding (s_e) for swapping
            arcface_model_for_swap = self.models_processor.get_arcface_model(parameters_for_face['SwapModelSelection'])
            s_e_for_swap_np = target_face_button.assigned_input_embedding.get(arcface_model_for_swap)
            
            if s_e_for_swap_np is None or \
               not isinstance(s_e_for_swap_np, np.ndarray) or \
               s_e_for_swap_np.size == 0 or \
               np.isnan(s_e_for_swap_np).any() or \
               np.isinf(s_e_for_swap_np).any():
                s_e_for_swap_np = None # Invalidate if problematic

            t_e_for_swap_np = target_face_button.get_embedding(arcface_model_for_swap) # Target embedding for likeness

            # DFM model instance
            dfm_model_name = parameters_for_face['DFMModelSelection']
            dfm_model_instance_local = None
            if parameters_for_face['SwapModelSelection'] == 'DeepFaceLive (DFM)' and dfm_model_name:
                dfm_model_instance_local = self.models_processor.load_dfm_model(dfm_model_name)

            # Proceed with swap_core if s_e is valid or if it's DFM mode with a valid instance
            if s_e_for_swap_np is not None or (parameters_for_face['SwapModelSelection'] == 'DeepFaceLive (DFM)' and dfm_model_instance_local):
                # swap_core expects the full image (perspective_crop here) and kps relative to it.
                # It returns a 512x512 swapped face.
                swapped_face_512_torch_rgb_uint8, comprehensive_mask_1x512x512_from_swap_core, _ = self.swap_core(
                    perspective_crop_torch_rgb_uint8, # The "full image" for this operation
                    face_kps_5_on_crop,             # Keypoints on this "full image"
                    s_e=s_e_for_swap_np,
                    t_e=t_e_for_swap_np,
                    parameters=parameters_for_face,
                    control=control_global,
                    dfm_model_instance=dfm_model_instance_local,
                    is_perspective_crop=True
                )
                
                # Paste the 512x512 swapped face back onto the perspective_crop_torch_rgb_uint8
                tform_persp_to_512template = self.get_face_similarity_tform(parameters_for_face['SwapModelSelection'], face_kps_5_on_crop)

                persp_border_mask_1x128x128 = self.get_border_mask(parameters_for_face)
                persp_final_combined_mask_1x512x512 = t512(persp_border_mask_1x128x128)
                persp_final_combined_mask_3x512x512_float = persp_final_combined_mask_1x512x512.repeat(3,1,1).float()

                # Use the comprehensive mask returned by swap_core
                if comprehensive_mask_1x512x512_from_swap_core is None or comprehensive_mask_1x512x512_from_swap_core.numel() == 0:
                    # Fallback to a full pass-through mask if something went wrong in swap_core's mask generation
                    persp_final_combined_mask_1x512x512_float_for_paste = torch.ones((1, 512, 512), dtype=torch.float32, device=perspective_crop_torch_rgb_uint8.device)
                else:
                    persp_final_combined_mask_1x512x512_float_for_paste = comprehensive_mask_1x512x512_from_swap_core.float() # Already 1x512x512 float

                persp_final_combined_mask_3x512x512_float_for_paste = persp_final_combined_mask_1x512x512_float_for_paste.repeat(3,1,1) # Ensure 3 channels

                masked_swapped_face_to_paste_float = swapped_face_512_torch_rgb_uint8.float() * persp_final_combined_mask_3x512x512_float_for_paste

                crop_h, crop_w = perspective_crop_torch_rgb_uint8.shape[1], perspective_crop_torch_rgb_uint8.shape[2]
                # get_grid_for_pasting needs a transform from target (persp_crop) to source (512_face)
                _, source_grid_normalized_xy_persp = self.get_grid_for_pasting(
                    tform_persp_to_512template, crop_h, crop_w, 512, 512, perspective_crop_torch_rgb_uint8.device
                )

                pasted_face_on_persp_float = torch.nn.functional.grid_sample(
                    masked_swapped_face_to_paste_float.unsqueeze(0),
                    source_grid_normalized_xy_persp,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False
                ).squeeze(0)

                transformed_mask_on_persp_float = torch.nn.functional.grid_sample(
                    persp_final_combined_mask_3x512x512_float_for_paste.unsqueeze(0),
                    source_grid_normalized_xy_persp,
                    mode='bilinear', padding_mode='zeros', align_corners=False
                ).squeeze(0)

                original_persp_crop_float = perspective_crop_torch_rgb_uint8.float()
                blended_persp_crop_float = pasted_face_on_persp_float + original_persp_crop_float * (1.0 - transformed_mask_on_persp_float)
                processed_crop_torch_rgb_uint8 = torch.clamp(blended_persp_crop_float, 0, 255).byte()
        
        return processed_crop_torch_rgb_uint8

    def process_frame(self, control: dict): # control is passed in
        # Input self.frame is HxWxC RGB uint8 NumPy array
        img_numpy_rgb_uint8 = self.frame
        
        # This tensor will be modified and eventually converted back to NumPy BGR
        processed_tensor_rgb_uint8 = torch.from_numpy(img_numpy_rgb_uint8).to(self.models_processor.device).permute(2,0,1)
        
        det_faces_data_for_display = [] # For overlays in standard mode

        if control.get('VR180ModeEnableToggle', False):
            # === VR180 Path ===
            # img_numpy_rgb_uint8 is HxWxC RGB uint8
            equirect_converter = EquirectangularConverter(
                img_numpy_rgb_uint8, device=self.models_processor.device
            )

            # Detect faces on the full equirectangular image to guide perspective cropping
            # run_detect expects CxHxW tensor
            bboxes_eq_np, _, _ = self.models_processor.run_detect( # kpss_5_eq_np not used directly here
                processed_tensor_rgb_uint8.clone(), # Use a clone for detection
                control['DetectorModelSelection'],
                max_num=control['MaxFacesToDetectSlider'],
                score=control['DetectorScoreSlider']/100.0,
                input_size=(img_numpy_rgb_uint8.shape[0], img_numpy_rgb_uint8.shape[1]), # H, W
                use_landmark_detection=control['LandmarkDetectToggle'],
                landmark_detect_mode=control['LandmarkDetectModelSelection'],
                landmark_score=control["LandmarkDetectScoreSlider"]/100.0,
                from_points=control["DetectFromPointsToggle"],
                rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270]
            )

            processed_perspective_crops_details = {}

            for i, bbox_eq_np_single in enumerate(bboxes_eq_np):
                # Determine eye based on horizontal position of bbox center in equirect image
                x_center_eq = (bbox_eq_np_single[0] + bbox_eq_np_single[2]) / 2
                eye_side = "L" if x_center_eq < equirect_converter.width / 2 else "R"
                
                # Process only one crop per eye for now to avoid redundant processing if multiple faces are in one eye's view
                if eye_side in processed_perspective_crops_details: continue

                # Use the currently selected target face in the UI for swapping
                selected_target_face_button = self.main_window.cur_selected_target_face_button
                if not selected_target_face_button: continue # No target selected, skip
                
                # Create ParametersDict for the current face in VR mode
                face_specific_params_vr = self.parameters.get(selected_target_face_button.face_id, {})
                default_params_dict_vr = dict(self.main_window.default_parameters) if isinstance(self.main_window.default_parameters, ParametersDict) else self.main_window.default_parameters
                if isinstance(face_specific_params_vr, ParametersDict): # Should be plain dict from self.parameters
                    face_specific_params_vr = dict(face_specific_params_vr)
                parameters_for_current_face_pd = ParametersDict(face_specific_params_vr, default_params_dict_vr)
                
                theta, phi = equirect_converter.calculate_theta_phi_from_bbox(bbox_eq_np_single)
                
                # Get perspective crop (returns Torch tensor CHW RGB uint8 on GPU)
                perspective_crop_torch_rgb_uint8 = equirect_converter.get_perspective_crop(
                    FOV=90, THETA=theta, PHI=phi, height=1024, width=1024 # Example size
                )
                if perspective_crop_torch_rgb_uint8 is None or perspective_crop_torch_rgb_uint8.numel() == 0:
                    print(f"VR180: Skipping empty perspective crop for eye {eye_side}")
                    continue
                
                # Only process (swap/edit) the crop if swapfacesButton or editFacesButton is checked
                if self.main_window.swapfacesButton.isChecked() or self.main_window.editFacesButton.isChecked():
                    processed_crop_torch_rgb_uint8 = self._process_single_vr_perspective_crop(
                        perspective_crop_torch_rgb_uint8,
                        selected_target_face_button,
                        parameters_for_current_face_pd,
                        control,
                        # eye_side_for_debug=f"_eye{eye_side}" # Pass if _process_single_vr_perspective_crop uses it
                    )
                else:
                    # If no swap/edit, use the original perspective crop
                    processed_crop_torch_rgb_uint8 = perspective_crop_torch_rgb_uint8

                
                processed_perspective_crops_details[eye_side] = {
                    'tensor_rgb_uint8': processed_crop_torch_rgb_uint8, # This is the processed crop
                    'theta': theta,
                    'phi': phi
                }
            
            # Stitch processed crops back
            # Start with the original equirect image as a Torch tensor
            # equirect_converter.equirect_tensor_cxhxw_rgb_uint8 is already CHW RGB uint8 on device
            final_equirect_torch_cxhxw_rgb_uint8 = equirect_converter.equirect_tensor_cxhxw_rgb_uint8.clone()
            
            p2e_converter = PerspectiveConverter(
                img_numpy_rgb_uint8, device=self.models_processor.device
            )

            for eye_side, data in processed_perspective_crops_details.items():
                # data['tensor_rgb_uint8'] is already CHW RGB uint8 Torch tensor on GPU
                p2e_converter.stitch_single_perspective(
                    target_equirect_torch_cxhxw_rgb_uint8=final_equirect_torch_cxhxw_rgb_uint8, # Modified in-place
                    processed_crop_torch_cxhxw_rgb_uint8=data['tensor_rgb_uint8'],
                    theta=data['theta'], phi=data['phi'], fov=90, # Must match FOV used for cropping
                    is_left_eye=(eye_side == "L")
                )
            
            processed_tensor_rgb_uint8 = final_equirect_torch_cxhxw_rgb_uint8


            # GPU Memory Cleanup for VR path
            if 'equirect_converter' in locals(): 
                for key in list(processed_perspective_crops_details.keys()): # Iterate over a copy of keys
                    if 'tensor_rgb_uint8' in processed_perspective_crops_details[key]:
                        del processed_perspective_crops_details[key]['tensor_rgb_uint8'] # Delete the tensor
                    # del processed_perspective_crops_details[key] # Deleting the inner dict entry                
                del equirect_converter
            if 'p2e_converter' in locals(): del p2e_converter
            #if 'final_equirect_torch_cxhxw_rgb_uint8' in locals(): del final_equirect_torch_cxhxw_rgb_uint8
            if 'processed_perspective_crops_details' in locals(): del processed_perspective_crops_details
            torch.cuda.empty_cache() # Use sparingly if memory issues persist
            
        else:
            # === Standard Path (adapting frame_worker-orig.py) ===
            # processed_tensor_rgb_uint8 is already the initial frame tensor (CxHxW RGB uint8)
            img_for_detection_and_swap = processed_tensor_rgb_uint8.clone()

            # Scaling logic from orig
            img_x = img_for_detection_and_swap.shape[2]
            img_y = img_for_detection_and_swap.shape[1]
            scale_applied_std = False
            if img_x < 512 and img_y < 512:
                if img_x <= img_y: new_h, new_w = int(512 * img_y / img_x), 512
                else: new_h, new_w = 512, int(512 * img_x / img_y)
                tscale = v2.Resize((new_h, new_w), antialias=True)
                img_for_detection_and_swap = tscale(img_for_detection_and_swap)
                scale_applied_std = True
            elif img_x < 512:
                new_h, new_w = int(512 * img_y / img_x), 512
                img_for_detection_and_swap = v2.Resize((new_h, new_w), antialias=True)(img_for_detection_and_swap)
                scale_applied_std = True
            elif img_y < 512:
                new_h, new_w = 512, int(512 * img_x / img_y)
                img_for_detection_and_swap = v2.Resize((new_h, new_w), antialias=True)(img_for_detection_and_swap)
                scale_applied_std = True

            # Rotation (applied before detection)
            if control['ManualRotationEnableToggle']:
                img_for_detection_and_swap = v2.functional.rotate(
                    img_for_detection_and_swap,
                    angle=control['ManualRotationAngleSlider'],
                    interpolation=v2.InterpolationMode.BILINEAR,
                    expand=True
                )

            # Face Detection
            use_landmark_detection = control['LandmarkDetectToggle']
            landmark_detect_mode = control['LandmarkDetectModelSelection']
            from_points = control["DetectFromPointsToggle"]
            if self.main_window.editFacesButton.isChecked(): # Force landmark settings for editor
                if not use_landmark_detection or landmark_detect_mode == "5":
                    use_landmark_detection = True
                    landmark_detect_mode = "203"
                from_points = True

            bboxes_std, kpss_5_std, kpss_all_std = self.models_processor.run_detect(
                img_for_detection_and_swap, # Detect on (potentially scaled and rotated) image
                control['DetectorModelSelection'],
                max_num=control['MaxFacesToDetectSlider'],
                score=control['DetectorScoreSlider']/100.0,
                input_size=(512, 512), # Hint for detector, actual input is img_for_detection_and_swap
                use_landmark_detection=use_landmark_detection,
                landmark_detect_mode=landmark_detect_mode,
                landmark_score=control["LandmarkDetectScoreSlider"]/100.0,
                from_points=from_points,
                rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270]
            )
            
            # Populate det_faces_data_for_display (used for overlays)
            # And also for iterating through faces to swap
            if len(kpss_5_std) > 0:
                for i in range(kpss_5_std.shape[0]):
                    face_kps_5 = kpss_5_std[i]
                    face_kps_all = kpss_all_std[i] if isinstance(kpss_all_std, np.ndarray) and kpss_all_std.ndim > 1 and i < kpss_all_std.shape[0] else face_kps_5
                    # Recognize face on the same image used for detection
                    face_emb, _ = self.models_processor.run_recognize_direct(img_for_detection_and_swap, face_kps_5, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                    det_faces_data_for_display.append({'kps_5': face_kps_5, 'kps_all': face_kps_all, 'embedding': face_emb, 'bbox': bboxes_std[i]})

            # Main processing loop for standard path
            if det_faces_data_for_display:
                for fface_data in det_faces_data_for_display:
                    # Determine which target face to use (e.g., currently selected or first one)
                    target_to_process_with = None
                    if self.main_window.cur_selected_target_face_button:
                        target_to_process_with = self.main_window.cur_selected_target_face_button
                    elif self.main_window.target_faces:
                        target_to_process_with = list(self.main_window.target_faces.values())[0]
                    
                    if not target_to_process_with: 
                        continue

                    # Create ParametersDict for the current face
                    face_specific_params = self.parameters.get(target_to_process_with.face_id, {})
                    default_params_dict = dict(self.main_window.default_parameters) if isinstance(self.main_window.default_parameters, ParametersDict) else self.main_window.default_parameters
                    if isinstance(face_specific_params, ParametersDict): # Should be plain dict
                        face_specific_params = dict(face_specific_params)
                    parameters_for_face_pd = ParametersDict(face_specific_params, default_params_dict)
                        
                    sim = self.models_processor.findCosineDistance(fface_data['embedding'], target_to_process_with.get_embedding(control['RecognitionModelSelection']))                   
                                   
                    if sim >= parameters_for_face_pd['SimilarityThresholdSlider']:
                        if self.main_window.swapfacesButton.isChecked() or self.main_window.editFacesButton.isChecked():
                            arcface_model_for_swap = self.models_processor.get_arcface_model(parameters_for_face_pd['SwapModelSelection'])
                            s_e_np = None
                            if self.main_window.swapfacesButton.isChecked(): # Only get s_e if actually swapping
                                s_e_np = target_to_process_with.assigned_input_embedding.get(arcface_model_for_swap)
                                if s_e_np is None or not isinstance(s_e_np, np.ndarray) or s_e_np.size == 0 or np.isnan(s_e_np).any() or np.isinf(s_e_np).any():
                                    s_e_np = None # Invalidate

                            t_e_np = target_to_process_with.get_embedding(arcface_model_for_swap)
                                
                            dfm_model_instance_local = None
                            if parameters_for_face_pd['SwapModelSelection'] == 'DeepFaceLive (DFM)':
                                dfm_model_name = parameters_for_face_pd('DFMModelSelection')
                                if dfm_model_name:
                                    dfm_model_instance_local = self.models_processor.load_dfm_model(dfm_model_name)
                            
                            # Proceed if s_e is valid (for latent models) or DFM is set up
                            if s_e_np is not None or (parameters_for_face_pd['SwapModelSelection'] == 'DeepFaceLive (DFM)' and dfm_model_instance_local is not None):
                                kps_5_adjusted = self.keypoints_adjustments(fface_data['kps_5'].copy(), parameters_for_face_pd) # Use copy
                                
                                # swap_core operates on img_for_detection_and_swap
                                img_for_detection_and_swap, original_face_comp, swap_mask_comp = self.swap_core(
                                    img_for_detection_and_swap,
                                    kps_5_adjusted,
                                    s_e=s_e_np, t_e=t_e_np,
                                    parameters=parameters_for_face_pd, control=control,
                                    dfm_model_instance=dfm_model_instance_local,
                                    is_perspective_crop=False # Standard path
                                )
                                fface_data['original_face'] = original_face_comp
                                fface_data['swap_mask'] = swap_mask_comp
                            
                        if self.main_window.editFacesButton.isChecked():
                            # swap_edit_face_core also operates on img_for_detection_and_swap
                            img_for_detection_and_swap = self.swap_edit_face_core(
                                img_for_detection_and_swap, fface_data['kps_all'], parameters_for_face_pd, control
                            )
            
            # Inverse Rotation (applied after all processing on img_for_detection_and_swap)
            if control['ManualRotationEnableToggle']:
                img_for_detection_and_swap = v2.functional.rotate(
                    img_for_detection_and_swap,
                    angle=-control['ManualRotationAngleSlider'], # Inverse angle
                    interpolation=v2.InterpolationMode.BILINEAR,
                    expand=True # Should match original expansion
                )
            
            # If scaling was applied, resize back to original dimensions of processed_tensor_rgb_uint8
            if scale_applied_std:
                original_h, original_w = processed_tensor_rgb_uint8.shape[1], processed_tensor_rgb_uint8.shape[2]
                processed_tensor_rgb_uint8 = v2.Resize((original_h, original_w), antialias=True)(img_for_detection_and_swap)
            else: # No scaling, so img_for_detection_and_swap is the final tensor
                processed_tensor_rgb_uint8 = img_for_detection_and_swap
        
        # --- Common Post-Processing (operates on processed_tensor_rgb_uint8) ---
        # Note: ManualRotation for VR output is not handled here.
        # The standard path rotation is self-contained (applied and then undone).

        if control['ShowAllDetectedFacesBBoxToggle'] and det_faces_data_for_display: # Only for std path
            processed_tensor_rgb_uint8 = self.draw_bounding_boxes_on_detected_faces(processed_tensor_rgb_uint8, det_faces_data_for_display, control)

        if control["ShowLandmarksEnableToggle"] and det_faces_data_for_display: # Only for std path
            temp_permuted = processed_tensor_rgb_uint8.permute(1,2,0) # HWC
            temp_permuted = self.paint_face_landmarks(temp_permuted, det_faces_data_for_display, control)
            processed_tensor_rgb_uint8 = temp_permuted.permute(2,0,1) # CHW

        compare_mode_active = self.is_view_face_mask or self.is_view_face_compare
        if compare_mode_active and det_faces_data_for_display: # Only for std path
             processed_tensor_rgb_uint8 = self.get_compare_faces_image(processed_tensor_rgb_uint8, det_faces_data_for_display, control)

        if control['FrameEnhancerEnableToggle'] and not compare_mode_active:
            processed_tensor_rgb_uint8 = self.enhance_core(processed_tensor_rgb_uint8, control=control)

        # Convert final tensor to NumPy BGR
        final_img_np_rgb_uint8 = processed_tensor_rgb_uint8.permute(1,2,0).cpu().numpy()
        if not final_img_np_rgb_uint8.flags['C_CONTIGUOUS']:
            final_img_np_rgb_uint8 = np.ascontiguousarray(final_img_np_rgb_uint8)
        
        return final_img_np_rgb_uint8[..., ::-1] # RGB to BGR

    def keypoints_adjustments(self, kps_5: np.ndarray, parameters: dict) -> np.ndarray:
        # This method modifies kps_5 in place if it's not a copy.
        # Ensure a copy is passed if original kps_5 needs to be preserved.
        kps_5_adj = kps_5.copy() 
        if parameters['FaceAdjEnableToggle']: 
            kps_5_adj[:,0] += parameters['KpsXSlider']
            kps_5_adj[:,1] += parameters['KpsYSlider']
            
            # Scaling from center (255,255) might be specific to a 512x512 assumption.
            # If kps are on a different sized image, this needs adjustment or a different center.
            # For now, assuming kps are in a space where this makes sense or it's handled by tform.
            kps_5_adj[:,0] -= 255 
            kps_5_adj[:,0] *= (1 + parameters['KpsScaleSlider'] / 100.0)
            kps_5_adj[:,0] += 255
            kps_5_adj[:,1] -= 255
            kps_5_adj[:,1] *= (1 + parameters['KpsScaleSlider'] / 100.0)
            kps_5_adj[:,1] += 255

        if parameters['LandmarksPositionAdjEnableToggle']:
            kps_5_adj[0][0] += parameters['EyeLeftXAmountSlider']
            kps_5_adj[0][1] += parameters['EyeLeftYAmountSlider']
            kps_5_adj[1][0] += parameters['EyeRightXAmountSlider']
            kps_5_adj[1][1] += parameters['EyeRightYAmountSlider']
            kps_5_adj[2][0] += parameters['NoseXAmountSlider']
            kps_5_adj[2][1] += parameters['NoseYAmountSlider']
            kps_5_adj[3][0] += parameters['MouthLeftXAmountSlider']
            kps_5_adj[3][1] += parameters['MouthLeftYAmountSlider']
            kps_5_adj[4][0] += parameters['MouthRightXAmountSlider']
            kps_5_adj[4][1] += parameters['MouthRightYAmountSlider']
        return kps_5_adj

    def paint_face_landmarks(self, img_hwc_rgb_uint8: torch.Tensor, det_faces_data: list, control: dict) -> torch.Tensor:
        # img_hwc_rgb_uint8 is HxWxC, Torch tensor on GPU
        img_hwc_rgb_uint8_out = img_hwc_rgb_uint8.clone() # Work on a clone
        point_thickness = 2 # Point thickness
        
        for fface_data in det_faces_data:
            # Determine parameters and keypoints to draw for this detected face
            # This logic assumes det_faces_data contains faces that might match a target_face
            # For simplicity, we'll use default color if no match or specific settings found.
            
            keypoints_to_draw = fface_data.get('kps_all') # Default to all keypoints
            landmark_color_rgb = (0, 255, 255) # Default color (Cyan for 'all')

            # Check if this detected face matches any target face to use specific settings
            matched_params = None
            for _, target_face_widget in self.main_window.target_faces.items():
                params_candidate = self.parameters.get(target_face_widget.face_id)
                if params_candidate:
                    sim = self.models_processor.findCosineDistance(
                        fface_data['embedding'],
                        target_face_widget.get_embedding(control['RecognitionModelSelection'])
                    )
                    if sim >= params_candidate['SimilarityThresholdSlider']: 
                        matched_params = params_candidate
                        break
            
            if matched_params and matched_params['LandmarksPositionAdjEnableToggle']:
                keypoints_to_draw = fface_data.get('kps_5')
                landmark_color_rgb = (255, 0, 0) # Red for adjusted 5 points
            
            if keypoints_to_draw is not None:
                for kpoint in keypoints_to_draw:
                    kx, ky = int(kpoint[0]), int(kpoint[1])
                    # Draw a small square for each keypoint
                    for i_offset in range(-point_thickness // 2, point_thickness // 2 + 1):
                        for j_offset in range(-point_thickness // 2, point_thickness // 2 + 1):
                            final_y, final_x = ky + i_offset, kx + j_offset
                            # Boundary checks
                            if 0 <= final_y < img_hwc_rgb_uint8_out.shape[0] and \
                               0 <= final_x < img_hwc_rgb_uint8_out.shape[1]:
                                img_hwc_rgb_uint8_out[final_y, final_x, 0] = landmark_color_rgb[0]
                                img_hwc_rgb_uint8_out[final_y, final_x, 1] = landmark_color_rgb[1]
                                img_hwc_rgb_uint8_out[final_y, final_x, 2] = landmark_color_rgb[2]
        return img_hwc_rgb_uint8_out

    def draw_bounding_boxes_on_detected_faces(self, img_cxhxw_rgb_uint8: torch.Tensor, det_faces_data: list, control: dict) -> torch.Tensor:
        # img_cxhxw_rgb_uint8 is CxHxW, Torch tensor on GPU
        img_out_cxhxw = img_cxhxw_rgb_uint8.clone() # Work on a clone
        
        for fface_data in det_faces_data:
            bbox = fface_data.get('bbox')
            if bbox is None: continue

            color_rgb = [0, 255, 0] # Green
            x_min, y_min, x_max, y_max = map(int, bbox)
            
            _, h, w = img_out_cxhxw.shape
            # Clamp coordinates to be within image bounds
            x_min_c, y_min_c = max(0, x_min), max(0, y_min)
            x_max_c, y_max_c = min(w - 1, x_max), min(h - 1, y_max)
            
            # Skip if bbox is invalid after clamping
            if x_min_c >= x_max_c or y_min_c >= y_max_c: continue

            max_dimension = max(h, w)
            thickness = max(1, max_dimension // 400) # Adjusted thickness, min 1
            
            color_tensor_c11 = torch.tensor(color_rgb, dtype=img_out_cxhxw.dtype, device=img_out_cxhxw.device).view(3, 1, 1)

            # Draw top edge
            img_out_cxhxw[:, y_min_c : min(y_min_c + thickness, y_max_c + 1), x_min_c : x_max_c + 1] = \
                color_tensor_c11.expand(-1, min(thickness, (y_max_c + 1) - y_min_c), x_max_c - x_min_c + 1)
            # Draw bottom edge
            img_out_cxhxw[:, max(y_min_c, y_max_c - thickness + 1) : y_max_c + 1, x_min_c : x_max_c + 1] = \
                color_tensor_c11.expand(-1, min(thickness, (y_max_c+1) - max(y_min_c, y_max_c - thickness + 1)), x_max_c - x_min_c + 1)
            # Draw left edge
            img_out_cxhxw[:, y_min_c : y_max_c + 1, x_min_c : min(x_min_c + thickness, x_max_c + 1) ] = \
                color_tensor_c11.expand(-1, y_max_c - y_min_c + 1, min(thickness, (x_max_c + 1) - x_min_c) )
            # Draw right edge
            img_out_cxhxw[:, y_min_c : y_max_c + 1, max(x_min_c, x_max_c - thickness + 1) : x_max_c + 1] = \
                color_tensor_c11.expand(-1, y_max_c - y_min_c + 1, min(thickness, (x_max_c+1) - max(x_min_c, x_max_c - thickness + 1)) )
        return img_out_cxhxw

    def get_compare_faces_image(self, img_cxhxw_rgb_uint8: torch.Tensor, det_faces_data: list, control: dict) -> torch.Tensor:
        # img_cxhxw_rgb_uint8 is CxHxW, Torch tensor on GPU
        imgs_to_vstack = []
        
        for fface_data in det_faces_data:
            # Check if this face matches a target face that has parameters for comparison
            target_face_match_found = False
            parameters_for_face = self.main_window.default_parameters # Fallback
            
            # Prefer currently selected target face if it matches
            if self.main_window.cur_selected_target_face_button:
                target_face = self.main_window.cur_selected_target_face_button
                params_candidate = self.parameters.get(target_face.face_id, self.main_window.default_parameters)
                sim = self.models_processor.findCosineDistance(
                    fface_data['embedding'],
                    target_face.get_embedding(control['RecognitionModelSelection'])
                )
                if sim >= params_candidate.get('SimilarityThresholdSlider', 0.5):
                    target_face_match_found = True
                    parameters_for_face = params_candidate
            
            # If no match with current selection, check all target faces (less ideal for compare view)
            if not target_face_match_found:
                for _, target_face_widget in self.main_window.target_faces.items():
                    face_specific_params_comp = self.parameters.get(target_face_widget.face_id, {})
                    default_params_dict_comp = dict(self.main_window.default_parameters) if isinstance(self.main_window.default_parameters, ParametersDict) else self.main_window.default_parameters
                    params_candidate = ParametersDict(face_specific_params_comp, default_params_dict_comp)
                    sim = self.models_processor.findCosineDistance(
                        fface_data['embedding'],
                        target_face_widget.get_embedding(control['RecognitionModelSelection'])
                    )
                    if sim >= params_candidate['SimilarityThresholdSlider']:
                        target_face_match_found = True
                        parameters_for_face = params_candidate
                        break # Take first match if multiple

            if target_face_match_found:
                # Get a 512x512 crop of the face from the main image using its keypoints
                # This is the "modified face" before enhancement for comparison purposes
                modified_face_512 = self.get_cropped_face_using_kps(img_cxhxw_rgb_uint8, fface_data['kps_5'], parameters_for_face)
                

                if control['FrameEnhancerEnableToggle']:
                    enhanced_version = self.enhance_core(modified_face_512.clone(), control=control)
                    # Ensure enhanced version is same size as modified_face_512 for cat
                    if enhanced_version.shape[1:] != modified_face_512.shape[1:]:
                        enhanced_version = v2.Resize(modified_face_512.shape[1:], antialias=True)(enhanced_version)
                    modified_face_512 = enhanced_version # Replace with enhanced if enabled
                
                imgs_to_cat_horizontally = []
                
                original_face_from_swap_core = fface_data.get('original_face') # This is HWC tensor
                if original_face_from_swap_core is not None:
                    imgs_to_cat_horizontally.append(original_face_from_swap_core.permute(2,0,1)) # CHW

                imgs_to_cat_horizontally.append(modified_face_512) # This is CHW

                swap_mask_from_swap_core = fface_data.get('swap_mask') # This is HWC tensor
                if swap_mask_from_swap_core is not None:
                    # Ensure it's 3-channel for concatenation if it's grayscale
                    mask_chw = swap_mask_from_swap_core.permute(2,0,1) # CHW
                    if mask_chw.shape[0] == 1: mask_chw = mask_chw.repeat(3,1,1)
                    imgs_to_cat_horizontally.append(mask_chw)
  
                if imgs_to_cat_horizontally:
                    # Ensure all tensors have same height for horizontal concatenation
                    min_h = min(t.shape[1] for t in imgs_to_cat_horizontally)
                    resized_imgs_to_cat = []
                    for t_img in imgs_to_cat_horizontally:
                        if t_img.shape[1] != min_h:
                            aspect_ratio = t_img.shape[2] / t_img.shape[1]
                            new_w = int(min_h * aspect_ratio)
                            resized_imgs_to_cat.append(v2.Resize((min_h, new_w), antialias=True)(t_img))
                        else:
                            resized_imgs_to_cat.append(t_img)
                    
                    img_compare_strip = torch.cat(resized_imgs_to_cat, dim=2) # Concatenate along width (dim=2)
                    imgs_to_vstack.append(img_compare_strip)
    
        if imgs_to_vstack:
            # Pad images to have the same width before vertical stacking
            max_width_for_vstack = max(img_strip.size(2) for img_strip in imgs_to_vstack)
            padded_strips_for_vstack = [
                torch.nn.functional.pad(img_strip, (0, max_width_for_vstack - img_strip.size(2), 0, 0)) # Pad width
                for img_strip in imgs_to_vstack
            ]
            # Stack image strips vertically
            final_comparison_image = torch.cat(padded_strips_for_vstack, dim=1) # Concatenate along height (dim=1)
            return final_comparison_image
        
        return img_cxhxw_rgb_uint8 # Return original if no comparisons generated
        
    def get_cropped_face_using_kps(self, img_cxhxw_rgb_uint8: torch.Tensor, kps_5: np.ndarray, parameters: dict) -> torch.Tensor:
        # img_cxhxw_rgb_uint8 is CxHxW, Torch tensor on GPU
        # kps_5 are keypoints relative to img_cxhxw_rgb_uint8
        tform = self.get_face_similarity_tform(parameters['SwapModelSelection'], kps_5)
        
        # Affine transform to get the 512x512 aligned face
        face_512_aligned = v2.functional.affine(
            img_cxhxw_rgb_uint8,
            angle=tform.rotation * 57.2958, # Radians to degrees
            translate=(tform.translation[0], tform.translation[1]),
            scale=tform.scale,
            shear=(0.0, 0.0), # No shear in SimilarityTransform
            center=(0,0), # Affine transform relative to origin
            interpolation=v2.InterpolationMode.BILINEAR
        )
        # Crop to 512x512 from the top-left of the transformed image
        face_512_cropped = v2.functional.crop(face_512_aligned, 0, 0, 512, 512)
        return face_512_cropped

    def get_face_similarity_tform(self, swapper_model: str, kps_5: np.ndarray) -> trans.SimilarityTransform:
        tform = trans.SimilarityTransform()
        # Determine destination points for alignment based on swapper model
        if swapper_model not in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS'):
            # Default ArcFace template for most models
            dst_points = faceutil.get_arcface_template(image_size=512, mode='arcface128')
            dst_points = np.squeeze(dst_points) # Ensure it's (5,2)
            tform.estimate(kps_5, dst_points)
        elif swapper_model == "CSCS":
            # CSCS uses FFHQ keypoints template
            # self.models_processor.FFHQ_kps should be loaded and available
            tform.estimate(kps_5, self.models_processor.FFHQ_kps)
        else: # GhostFace models
            dst_points = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            # GhostFace uses a specific matrix estimation
            M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst_points)
            tform.params[0:2] = M # Directly set the 2x3 matrix part of SimilarityTransform
        return tform
      
    def get_transformed_and_scaled_faces(self, tform: trans.SimilarityTransform, img_cxhxw_rgb_uint8: torch.Tensor) -> tuple:
        # img_cxhxw_rgb_uint8 is CxHxW, Torch tensor on GPU
        # tform maps from img_cxhxw_rgb_uint8 space to the canonical 512x512 template
        
        original_face_512 = v2.functional.affine(
            img_cxhxw_rgb_uint8,
            angle=tform.rotation * 57.2958, # Radians to degrees
            translate=(tform.translation[0], tform.translation[1]),
            scale=tform.scale,
            shear=(0.0, 0.0),
            center=(0,0),
            interpolation=v2.InterpolationMode.BILINEAR
        )
        original_face_512 = v2.functional.crop(original_face_512, 0, 0, 512, 512) # Crop to 512x512
        
        original_face_384 = t384(original_face_512)
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256) # Note: original code had t128(original_face_256)
        
        return original_face_512, original_face_384, original_face_256, original_face_128
    
    def get_affined_face_dim_and_swapping_latents(self, original_faces: tuple, swapper_model: str,
                                                  dfm_model_name_from_ui: str,
                                                  s_e: np.ndarray | None, t_e: np.ndarray | None,
                                                  parameters: dict):
        # original_faces: (face_512, face_384, face_256, face_128) all CxHxW uint8
        original_face_512, original_face_384, original_face_256, original_face_128 = original_faces
        
        input_face_affined = None # This will be CxHxW uint8
        dfm_model_instance = None
        dim = 1 # Corresponds to 128x128 base tile size for Inswapper
        latent = None # Torch tensor

        # --- Pre-load necessary components based on swapper_model ---
        if swapper_model == 'Inswapper128':
            self.models_processor.load_inswapper_iss_emap('Inswapper128')
        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            self.models_processor.load_inswapper_iss_emap(swapper_model)
        # Other models (SimSwap, Ghost, CSCS, DFM) might have their own loading mechanisms.

        # --- Calculate latent if s_e is valid ---
        if s_e is not None: # s_e is already checked for None/NaN/Inf by caller
            calc_latent_fn_map = {
                'Inswapper128': self.models_processor.calc_inswapper_latent,
                'InStyleSwapper256 Version A': lambda emb: self.models_processor.calc_swapper_latent_iss(emb, 'A'),
                'InStyleSwapper256 Version B': lambda emb: self.models_processor.calc_swapper_latent_iss(emb, 'B'),
                'InStyleSwapper256 Version C': lambda emb: self.models_processor.calc_swapper_latent_iss(emb, 'C'),
                'SimSwap512': self.models_processor.calc_swapper_latent_simswap512,
                'GhostFace-v1': self.models_processor.calc_swapper_latent_ghost,
                'GhostFace-v2': self.models_processor.calc_swapper_latent_ghost,
                'GhostFace-v3': self.models_processor.calc_swapper_latent_ghost,
                'CSCS': self.models_processor.calc_swapper_latent_cscs
            }
            calc_latent_fn = calc_latent_fn_map.get(swapper_model)

            if calc_latent_fn:
                s_e_latent_np = calc_latent_fn(s_e)
                if np.isnan(s_e_latent_np).any() or np.isinf(s_e_latent_np).any():
                    # This should ideally not happen if s_e was pre-validated, but good to check calc output
                    return None, None, dim, None # Error state
                latent = torch.from_numpy(s_e_latent_np).float().to(self.models_processor.device)

                if parameters['FaceLikenessEnableToggle'] and t_e is not None: # t_e also pre-validated
                    factor = parameters['FaceLikenessFactorDecimalSlider']
                    dst_latent_np = calc_latent_fn(t_e)
                    if not (np.isnan(dst_latent_np).any() or np.isinf(dst_latent_np).any()):
                        dst_latent_torch = torch.from_numpy(dst_latent_np).float().to(self.models_processor.device)
                        if not (torch.isnan(dst_latent_torch).any() or torch.isinf(dst_latent_torch).any()):
                            latent = latent - (factor * dst_latent_torch)
            # If swapper_model is DFM, latent remains None (or empty list as per original)
            elif swapper_model == 'DeepFaceLive (DFM)':
                latent = [] # DFM doesn't use s_e/t_e for latent in this way

        # --- Determine input_face_affined and dim ---
        # And load DFM model if needed
        if swapper_model == 'Inswapper128':
            res_selection = parameters['SwapperResSelection']
            if res_selection == '128': dim, input_face_affined = 1, original_face_128
            elif res_selection == '256': dim, input_face_affined = 2, original_face_256
            elif res_selection == '384': dim, input_face_affined = 3, original_face_384
            elif res_selection == '512': dim, input_face_affined = 4, original_face_512
            else: dim, input_face_affined = 1, original_face_128 # Default
        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            dim, input_face_affined = 2, original_face_256
        elif swapper_model == 'SimSwap512':
            dim, input_face_affined = 4, original_face_512
        elif swapper_model in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS'):
            dim, input_face_affined = 2, original_face_256
        elif swapper_model == 'DeepFaceLive (DFM)':
            if dfm_model_name_from_ui:
                dfm_model_instance = self.models_processor.load_dfm_model(dfm_model_name_from_ui)
                if not dfm_model_instance: return None, None, 4, latent # DFM load failed
            else: # DFM selected but no model name
                return None, None, 4, latent # Error state
            input_face_affined = original_face_512
            dim = 4 # DFM effectively works on 512x512 (or its internal size)
        else: # Unknown swapper model
            return None, None, dim, latent # Error state

        # --- Apply FaceAdjEnableToggle scaling to the chosen input_face_affined ---
        if input_face_affined is not None and parameters['FaceAdjEnableToggle']:
            scale_factor = 1.0 + parameters['FaceScaleAmountSlider'] / 100.0
            if abs(scale_factor - 1.0) > 1e-6: # Only apply if scale changes
                h, w = input_face_affined.shape[1], input_face_affined.shape[2]
                center_coords = (w / 2.0, h / 2.0)
                input_face_affined = v2.functional.affine(
                    input_face_affined, angle=0.0, translate=(0.0, 0.0),
                    scale=scale_factor, shear=(0.0, 0.0),
                    center=center_coords, interpolation=v2.InterpolationMode.BILINEAR
                )
        return input_face_affined, dfm_model_instance, dim, latent

    def get_swapped_and_prev_face(self, output_placeholder_hwc_float: torch.Tensor,
                                  input_face_affined_hwc_float: torch.Tensor, # HxWxC, range [0,1] (target face for swapper)
                                  original_face_512_cxhxw_uint8: torch.Tensor, # Cx512x512 uint8 (used by DFM)
                                  latent: torch.Tensor | list | None, itex: int, dim: int, swapper_model: str,
                                  dfm_model_instance, parameters: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Returns: (swapped_512_cxhxw_uint8, prev_face_hwc_float_for_strength_blend)
        # prev_face_hwc_float_for_strength_blend is HxWxC float [0,1], matching input_face_affined_hwc_float dimensions

        # Fallback if latent is problematic (already checked by caller, but as a safeguard)
        if latent is not None and isinstance(latent, torch.Tensor) and (torch.isnan(latent).any() or torch.isinf(latent).any()):
            # Fallback: return original face (resized to 512x512 uint8) and the input affine face as prev_face
            swapped_512_uint8 = t512( (input_face_affined_hwc_float.permute(2,0,1) * 255.0).byte() )
            return swapped_512_uint8, input_face_affined_hwc_float.clone()

        prev_face_hwc_float_for_strength_blend = input_face_affined_hwc_float.clone() # For strength blend
        current_iter_face_hwc_float = input_face_affined_hwc_float.clone() # For iterative swapping

        # This will hold the output of the swapper model in its native resolution, float [0,1] RGB
        swapped_face_native_res_cxhxw_float = current_iter_face_hwc_float.permute(2,0,1) # Default to input

        # --- Model-specific swapping logic ---
        # Each block should update swapped_face_native_res_cxhxw_float and prev_face_hwc_float_for_strength_blend
        
        current_input_cxhxw_float = current_iter_face_hwc_float.permute(2,0,1) # CxHxW, [0,1]

        if swapper_model == 'Inswapper128':
            # input_face_affined is HWC, float [0,1] (current state for this iteration)
            # output is HWC, float [0,1] (will store result of this iteration)
            with torch.no_grad():
                for _ in range(itex): # Iterative refinement
                    for j in range(dim): # Original loop variable j
                        for i in range(dim): # Original loop variable i
                            # Strided slicing from input_face_affined_hwc_float (HWC, float [0,1])
                            input_tile_hwc_0_1 = input_face_affined_hwc_float[j::dim, i::dim, :] 
                            
                            # Convert to BCHW, float [0,1] for run_inswapper (as per original logic)
                            input_tile_bchw_0_1 = input_tile_hwc_0_1.permute(2, 0, 1).unsqueeze(0).contiguous()                            
                            
                            # Assuming run_inswapper takes [0,1] and outputs [0,1] (BCHW)
                            # This matches the behavior of the original frame_worker-orig.py.
                            swapper_output_tile_bchw_0_1 = torch.empty((1,3,128,128), dtype=torch.float32, device=self.models_processor.device).contiguous()
                            self.models_processor.run_inswapper(input_tile_bchw_0_1, latent, swapper_output_tile_bchw_0_1)

                            # Convert output to HWC, float [0,1]
                            swapper_output_tile_hwc_0_1 = torch.squeeze(swapper_output_tile_bchw_0_1).permute(1, 2, 0) 
                            
                            output_placeholder_hwc_float[j::dim, i::dim, :] = swapper_output_tile_hwc_0_1.clone()
                    
                    if torch.isnan(output_placeholder_hwc_float).any() or torch.isinf(output_placeholder_hwc_float).any(): break # Error in iteration
                    
                    prev_face_hwc_float_for_strength_blend = input_face_affined_hwc_float.clone() # Save HWC [0,1]

                    input_face_affined_hwc_float = output_placeholder_hwc_float.clone() # Update for next iteration HWC [0,1]
            
            swapped_face_native_res_cxhxw_float = input_face_affined_hwc_float.permute(2,0,1) # Final result CHW, [0,1]
            
        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1]
            with torch.no_grad():
                for _ in range(itex):
                    # ISS expects input [0,1]
                    swapper_output_256_batched = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device)
                    self.models_processor.run_iss_swapper(current_input_cxhxw_float.unsqueeze(0), latent, swapper_output_256_batched, version)
                    model_raw_output_cxhxw_float_0_1 = swapper_output_256_batched.squeeze(0)
                    
                    if torch.isnan(model_raw_output_cxhxw_float_0_1).any() or torch.isinf(model_raw_output_cxhxw_float_0_1).any():
                        break
                        
                    prev_face_hwc_float_for_strength_blend = current_iter_face_hwc_float.clone()
                    current_iter_face_hwc_float = model_raw_output_cxhxw_float_0_1.permute(1,2,0) # Already [0,1] HWC
                    current_input_cxhxw_float = current_iter_face_hwc_float.permute(2,0,1)
                    swapped_face_native_res_cxhxw_float = current_input_cxhxw_float

        elif swapper_model == 'SimSwap512':
            with torch.no_grad():
                for _ in range(itex):
                    # SimSwap expects input [0,1]
                    swapper_output_512_batched = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device)
                    self.models_processor.run_swapper_simswap512(current_input_cxhxw_float.unsqueeze(0), latent, swapper_output_512_batched)
                    model_raw_output_cxhxw_float_0_1 = swapper_output_512_batched.squeeze(0)

                    if torch.isnan(model_raw_output_cxhxw_float_0_1).any() or torch.isinf(model_raw_output_cxhxw_float_0_1).any():
                        break

                    prev_face_hwc_float_for_strength_blend = current_iter_face_hwc_float.clone()
                    current_iter_face_hwc_float = model_raw_output_cxhxw_float_0_1.permute(1,2,0) # Already [0,1] HWC
                    current_input_cxhxw_float = current_iter_face_hwc_float.permute(2,0,1)
                    swapped_face_native_res_cxhxw_float = current_input_cxhxw_float
        
        elif swapper_model in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3'):
            input_bgr_cxhxw_float = current_input_cxhxw_float[[2,1,0], :, :] # RGB to BGR
            input_normalized_bgr_cxhxw_float = (input_bgr_cxhxw_float * 2.0) - 1.0 # To [-1,1]
            with torch.no_grad():
                for _ in range(itex):
                    swapper_output_256_bgr_batched = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device)
                    self.models_processor.run_swapper_ghostface(input_normalized_bgr_cxhxw_float.unsqueeze(0), latent, swapper_output_256_bgr_batched, swapper_model)
                    model_raw_output_bgr_cxhxw_float_neg1_1 = swapper_output_256_bgr_batched.squeeze(0)

                    if torch.isnan(model_raw_output_bgr_cxhxw_float_neg1_1).any() or torch.isinf(model_raw_output_bgr_cxhxw_float_neg1_1).any():
                        break
                    
                    model_raw_output_rgb_cxhxw_float_neg1_1 = model_raw_output_bgr_cxhxw_float_neg1_1[[2,1,0], :, :] # BGR to RGB
                    
                    prev_face_hwc_float_for_strength_blend = current_iter_face_hwc_float.clone()
                    current_iter_face_hwc_float = ((model_raw_output_rgb_cxhxw_float_neg1_1 + 1.0) / 2.0).permute(1,2,0) # To [0,1] HWC
                    
                    current_input_cxhxw_float = current_iter_face_hwc_float.permute(2,0,1)
                    input_bgr_cxhxw_float = current_input_cxhxw_float[[2,1,0], :, :]
                    input_normalized_bgr_cxhxw_float = (input_bgr_cxhxw_float * 2.0) - 1.0
                    swapped_face_native_res_cxhxw_float = current_input_cxhxw_float

        elif swapper_model == 'CSCS':
            input_normalized_cxhxw_float = (current_input_cxhxw_float - 0.5) / 0.5 # Normalize with (0.5,0.5,0.5), (0.5,0.5,0.5) -> [-1,1]
            with torch.no_grad():
                for _ in range(itex):
                    swapper_output_256_batched = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device)
                    self.models_processor.run_swapper_cscs(input_normalized_cxhxw_float.unsqueeze(0), latent, swapper_output_256_batched)
                    model_raw_output_cxhxw_float_neg1_1 = swapper_output_256_batched.squeeze(0) # Output is also normalized

                    if torch.isnan(model_raw_output_cxhxw_float_neg1_1).any() or torch.isinf(model_raw_output_cxhxw_float_neg1_1).any():
                        break
                    
                    model_raw_output_cxhxw_float_0_1 = (model_raw_output_cxhxw_float_neg1_1 * 0.5) + 0.5 # Denormalize to [0,1]

                    prev_face_hwc_float_for_strength_blend = current_iter_face_hwc_float.clone()
                    current_iter_face_hwc_float = model_raw_output_cxhxw_float_0_1.permute(1,2,0) # To [0,1] HWC
                    current_input_cxhxw_float = current_iter_face_hwc_float.permute(2,0,1)
                    input_normalized_cxhxw_float = (current_input_cxhxw_float - 0.5) / 0.5
                    swapped_face_native_res_cxhxw_float = current_input_cxhxw_float

        elif swapper_model == 'DeepFaceLive (DFM)' and dfm_model_instance:
            # DFM takes Cx512x512 uint8 and returns Cx512x512 uint8
            if original_face_512_cxhxw_uint8 is None or original_face_512_cxhxw_uint8.numel() == 0:
                # Fallback if DFM input is bad
                swapped_face_native_res_cxhxw_float = current_input_cxhxw_float
            else:
                out_celeb_cxhxw_uint8, _, _ = dfm_model_instance.convert(
                    original_face_512_cxhxw_uint8,
                    parameters.get('DFMAmpMorphSlider', 0)/100.0,
                    rct=parameters.get('DFMRCTColorToggle', False)
                )
                if torch.isnan(out_celeb_cxhxw_uint8.float()).any() or torch.isinf(out_celeb_cxhxw_uint8.float()).any():
                     swapped_face_native_res_cxhxw_float = current_input_cxhxw_float # Fallback
                else:
                    swapped_face_native_res_cxhxw_float = out_celeb_cxhxw_uint8.float() / 255.0 # Convert to [0,1] float
            # For DFM, prev_face for strength blend is less direct if iterations were intended.
            # Here, we assume itex=1 for DFM or strength blend is handled differently.
            # prev_face_hwc_float_for_strength_blend remains the initial input_face_affined_hwc_float
        
        # Final conversion to 512x512 uint8
        # Ensure swapped_face_native_res_cxhxw_float is valid before t512
        if torch.isnan(swapped_face_native_res_cxhxw_float).any() or torch.isinf(swapped_face_native_res_cxhxw_float).any():
            # If error during swap, use the input affine face as the result
            swapped_face_native_res_cxhxw_float = input_face_affined_hwc_float.permute(2,0,1)
            # Ensure prev_face is also safe for strength blending if it was corrupted
            if prev_face_hwc_float_for_strength_blend is None or \
               torch.isnan(prev_face_hwc_float_for_strength_blend).any() or \
               torch.isinf(prev_face_hwc_float_for_strength_blend).any():
                prev_face_hwc_float_for_strength_blend = input_face_affined_hwc_float.clone()


        resized_swapped_face_float_0_1 = t512(swapped_face_native_res_cxhxw_float) # Resize to 512x512
        swapped_512_cxhxw_uint8 = (torch.clamp(resized_swapped_face_float_0_1 * 255.0, 0, 255)).byte()
        
        return swapped_512_cxhxw_uint8, prev_face_hwc_float_for_strength_blend
            
    def get_border_mask(self, parameters: dict) -> torch.Tensor:
        # Returns 1x128x128 float tensor mask
        border_mask = torch.ones((1, 128, 128), dtype=torch.float32, device=self.models_processor.device)

        top = parameters['BorderTopSlider']
        left = parameters['BorderLeftSlider']
        right = 128 - parameters['BorderRightSlider']
        bottom = 128 - parameters['BorderBottomSlider']

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        blur_amount = parameters['BorderBlurSlider']
        blur_kernel_size = blur_amount * 2 + 1
        if blur_kernel_size > 1:
            # Ensure sigma is positive and reasonable
            sigma_val = max(blur_amount * 0.15 + 0.1, 1e-6) 
            gauss = transforms.GaussianBlur(blur_kernel_size, sigma=sigma_val)
            border_mask = gauss(border_mask)
        return border_mask
            
    def swap_core(self, img_cxhxw_rgb_uint8: torch.Tensor, kps_5: np.ndarray,
                  s_e: np.ndarray | None = None, t_e: np.ndarray | None = None,
                  parameters: dict | None = None, control: dict | None = None,
                  dfm_model_instance=None,
                  is_perspective_crop: bool = False
                  ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # img_cxhxw_rgb_uint8: CxHxW RGB uint8 tensor (the image to paste onto)
        # kps_5: 5 keypoints for the face in img_cxhxw_rgb_uint8
        # Returns:
        #   - If is_perspective_crop=True: (swapped_face_512_cxhxw_uint8, None, None)
        #   - If is_perspective_crop=False: (img_cxhxw_rgb_uint8_with_pasted_face, original_face_for_compare_tensor, swap_mask_for_compare_tensor)
        if parameters is None: # Should be ParametersDict
            # Fallback if None, though callers should ensure a ParametersDict is passed
            default_params_dict = dict(self.main_window.default_parameters) if isinstance(self.main_window.default_parameters, ParametersDict) else self.main_window.default_parameters
            parameters = ParametersDict({}, default_params_dict)

        control = control if control is not None else {}
        swapper_model = parameters['SwapModelSelection']
        
        # Validate s_e and t_e early
        valid_s_e = None
        if s_e is not None and isinstance(s_e, np.ndarray) and s_e.size > 0 and \
           not (np.isnan(s_e).any() or np.isinf(s_e).any()):
            valid_s_e = s_e
        
        valid_t_e = None
        if t_e is not None and isinstance(t_e, np.ndarray) and t_e.size > 0 and \
           not (np.isnan(t_e).any() or np.isinf(t_e).any()):
            valid_t_e = t_e

        # Get transformation matrix for aligning the face
        tform = self.get_face_similarity_tform(swapper_model, kps_5)
        
        # Get aligned 512x512 version of the original face, and other scales
        original_faces_tuple = self.get_transformed_and_scaled_faces(tform, img_cxhxw_rgb_uint8)
        original_face_512_cxhxw_uint8, _, original_face_256_for_masks, _ = original_faces_tuple

        # This will hold the 512x512 swapped face after model processing
        swapped_final_512_cxhxw_uint8 = original_face_512_cxhxw_uint8.clone() # Default to original
        prev_face_for_strength_blend_hwc_float = None # HxWxC float [0,1]

        # Condition to attempt swap: valid_s_e must exist OR it's DFM mode with a valid instance
        attempt_swap = (valid_s_e is not None) or \
                       (swapper_model == 'DeepFaceLive (DFM)' and dfm_model_instance is not None)

        if attempt_swap:
            # Get the appropriately scaled input face for the swapper model, and the latent vector
            input_face_affined_cxhxw_uint8, local_dfm_instance_from_getter, dim, latent = \
                self.get_affined_face_dim_and_swapping_latents(
                    original_faces_tuple, swapper_model,
                    parameters['DFMModelSelection'], # DFM model name from UI
                    valid_s_e, valid_t_e, parameters
                )
            
            # Use pre-loaded dfm_model_instance if provided, otherwise use one from getter (if DFM)
            current_dfm_instance_for_swap = dfm_model_instance if dfm_model_instance is not None else local_dfm_instance_from_getter

            # Further checks: if get_affined_face_dim_and_swapping_latents returned None for critical parts
            can_proceed_with_model_run = False
            if swapper_model == 'DeepFaceLive (DFM)':
                can_proceed_with_model_run = current_dfm_instance_for_swap is not None and \
                                             input_face_affined_cxhxw_uint8 is not None
            else: # Latent-based models
                can_proceed_with_model_run = latent is not None and \
                                             input_face_affined_cxhxw_uint8 is not None and \
                                             not (torch.isnan(latent).any() or torch.isinf(latent).any())
            
            if can_proceed_with_model_run:
                itex = 1
                if parameters['StrengthEnableToggle']: # Direct access
                    itex = ceil(parameters['StrengthAmountSlider'] / 100.0) # Direct access
                    itex = max(1, itex) # Iterations should be at least 1 if strength is on

                # Prepare inputs for get_swapped_and_prev_face
                # output_placeholder is HxWxC float, matching input_face_affined's dimensions
                output_placeholder_hwc_float = torch.zeros_like(
                    input_face_affined_cxhxw_uint8.permute(1,2,0), dtype=torch.float32
                )
                input_face_affined_hwc_float = (input_face_affined_cxhxw_uint8.float() / 255.0).permute(1,2,0)

                swapped_final_512_cxhxw_uint8, prev_face_for_strength_blend_hwc_float = \
                    self.get_swapped_and_prev_face(
                        output_placeholder_hwc_float, input_face_affined_hwc_float,
                        original_face_512_cxhxw_uint8, # For DFM
                        latent, itex, dim, swapper_model,
                        current_dfm_instance_for_swap, parameters
                    )
        
        # --- Post-swapper model processing (strength, restorers, color, etc.) ---
        if parameters['StrengthEnableToggle'] and prev_face_for_strength_blend_hwc_float is not None and \
           not (torch.isnan(prev_face_for_strength_blend_hwc_float).any() or torch.isinf(prev_face_for_strength_blend_hwc_float).any()):
            
            itex_for_strength = ceil(parameters['StrengthAmountSlider'] / 100.0)
            itex_for_strength = max(0, itex_for_strength) # Can be 0 if slider is 0

            if itex_for_strength == 0: # Strength is 0%, use original face
                swapped_final_512_cxhxw_uint8 = original_face_512_cxhxw_uint8.clone()
            else:
                alpha = np.mod(parameters['StrengthAmountSlider'], 100) * 0.01
                if alpha == 0 and parameters['StrengthAmountSlider'] > 0 : alpha = 1.0

                # prev_face_for_strength_blend_hwc_float is HxWxC, [0,1]
                # Needs to be Cx512x512 uint8 for blending with swapped_final_512_cxhxw_uint8
                prev_face_cxhxw_float_0_1 = prev_face_for_strength_blend_hwc_float.permute(2,0,1)
                prev_face_512_cxhxw_uint8_for_blend = (torch.clamp(t512(prev_face_cxhxw_float_0_1) * 255.0, 0, 255)).byte()
                
                swapped_final_512_cxhxw_uint8 = (
                    swapped_final_512_cxhxw_uint8.float() * alpha +
                    prev_face_512_cxhxw_uint8_for_blend.float() * (1.0 - alpha)
                ).byte()
        
        border_mask_1x128x128 = self.get_border_mask(parameters)
        current_composite_mask_1x128x128 = torch.ones((1, 128, 128), dtype=torch.float32, device=self.models_processor.device)

        if parameters['FaceExpressionEnableToggle']:
            swapped_final_512_cxhxw_uint8 = self.apply_face_expression_restorer(original_face_512_cxhxw_uint8, swapped_final_512_cxhxw_uint8, parameters)

        if parameters["FaceRestorerEnableToggle"]:
            swapped_final_512_cxhxw_uint8 = self.models_processor.apply_facerestorer(swapped_final_512_cxhxw_uint8, parameters['FaceRestorerDetTypeSelection'], parameters['FaceRestorerTypeSelection'], parameters["FaceRestorerBlendSlider"], parameters['FaceFidelityWeightDecimalSlider'], control['DetectorScoreSlider']/100.0)
        
        if parameters["FaceRestorerEnable2Toggle"]:
            swapped_final_512_cxhxw_uint8 = self.models_processor.apply_facerestorer(swapped_final_512_cxhxw_uint8, parameters['FaceRestorerDetType2Selection'], parameters['FaceRestorerType2Selection'], parameters["FaceRestorerBlend2Slider"], parameters['FaceFidelityWeight2DecimalSlider'], control['DetectorScoreSlider']/100.0)

        if parameters["OccluderEnableToggle"]:
            mask = self.models_processor.apply_occlusion(original_face_256_for_masks, parameters["OccluderSizeSlider"])
            mask = t128(mask)
            swap_mask = torch.mul(current_composite_mask_1x128x128, mask)
            gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
            current_composite_mask_1x128x128 = gauss(swap_mask)
        
        if parameters["DFLXSegEnableToggle"]:
            img_mask = self.models_processor.apply_dfl_xseg(original_face_256_for_masks, -parameters["DFLXSegSizeSlider"])
            img_mask = t128(img_mask) # Resize to 128x128
            current_composite_mask_1x128x128 = torch.mul(current_composite_mask_1x128x128, 1.0 - img_mask) # Invert DFLXSeg mask
            dfx_blur_val = parameters['OccluderXSegBlurSlider'] * 2 + 1
            if dfx_blur_val > 1:
                sigma_dfx = max(parameters['OccluderXSegBlurSlider']*0.15 + 0.1, 1e-6)
                current_composite_mask_1x128x128 = transforms.GaussianBlur(dfx_blur_val, sigma=sigma_dfx)(current_composite_mask_1x128x128)

        if parameters["FaceParserEnableToggle"]:
            mask = self.models_processor.apply_face_parser(swapped_final_512_cxhxw_uint8, parameters)
            mask = t128(mask) # Resize to 128x128
            current_composite_mask_1x128x128 = torch.mul(current_composite_mask_1x128x128, mask)

        if parameters["ClipEnableToggle"]:
            mask = self.models_processor.run_CLIPs(original_face_512_cxhxw_uint8, parameters["ClipText"], parameters["ClipAmountSlider"])
            mask = t128(mask) # Resize to 128x128
            current_composite_mask_1x128x128 *= mask

        if parameters['RestoreMouthEnableToggle'] or parameters['RestoreEyesEnableToggle']:
            # Keypoints need to be in the 512x512 aligned face space
            ones_column = np.ones((kps_5.shape[0], 1), dtype=np.float32)
            homogeneous_kps_on_orig_img = np.hstack([kps_5, ones_column])
            # tform.params[0:2] is the 2x3 matrix from original image to 512 template
            dst_kps_5_on_aligned_512 = np.dot(homogeneous_kps_on_orig_img, tform.params[0:2].T)

            img_swap_mask_1x512x512 = torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)

            if parameters['RestoreMouthEnableToggle']:
                img_swap_mask_1x512x512 = self.models_processor.restore_mouth(
                    torch.zeros_like(img_swap_mask_1x512x512), # orig_mask_tensor (not really used)
                    img_swap_mask_1x512x512, dst_kps_5_on_aligned_512,
                    parameters['RestoreMouthBlendAmountSlider']/100.0,
                    parameters['RestoreMouthFeatherBlendSlider'],
                    parameters['RestoreMouthSizeFactorSlider']/100.0,
                    parameters['RestoreXMouthRadiusFactorDecimalSlider'],
                    parameters['RestoreYMouthRadiusFactorDecimalSlider'],
                    parameters['RestoreXMouthOffsetSlider'],
                    parameters['RestoreYMouthOffsetSlider']
                )

            if parameters['RestoreEyesEnableToggle']:
                img_swap_mask_1x512x512 = self.models_processor.restore_eyes(
                    torch.zeros_like(img_swap_mask_1x512x512), # orig_mask_tensor
                    img_swap_mask_1x512x512, dst_kps_5_on_aligned_512,
                    parameters['RestoreEyesBlendAmountSlider']/100.0,
                    parameters['RestoreEyesFeatherBlendSlider'],
                    parameters['RestoreEyesSizeFactorDecimalSlider'],
                    parameters['RestoreXEyesRadiusFactorDecimalSlider'],
                    parameters['RestoreYEyesRadiusFactorDecimalSlider'],
                    parameters['RestoreXEyesOffsetSlider'],
                    parameters['RestoreYEyesOffsetSlider'],
                    parameters['RestoreEyesSpacingOffsetSlider']
                )
            
            img_swap_mask_1x512x512 = torch.clamp(img_swap_mask_1x512x512, 0, 1)
            rem_blur_val = parameters['RestoreEyesMouthBlurSlider']*2+1

            if rem_blur_val > 1:
                 sigma_rem = max(parameters['RestoreEyesMouthBlurSlider']*0.15 + 0.1, 1e-6)
                 img_swap_mask_1x512x512 = transforms.GaussianBlur(rem_blur_val, sigma=sigma_rem)(img_swap_mask_1x512x512)
            
            img_swap_mask_1x128x128 = t128(img_swap_mask_1x512x512)
            current_composite_mask_1x128x128 = torch.mul(current_composite_mask_1x128x128, img_swap_mask_1x128x128)

        # --- Post-masking color adjustments and effects ---
        if parameters["DifferencingEnableToggle"]:
            mask_diff = self.models_processor.apply_fake_diff(swapped_final_512_cxhxw_uint8, original_face_512_cxhxw_uint8, parameters["DifferencingAmountSlider"])
            diff_blur_val = parameters['DifferencingBlendAmountSlider']*2+1
            if diff_blur_val > 1:
                 sigma_diff = max(parameters['DifferencingBlendAmountSlider']*0.15 + 0.1, 1e-6)
                 mask_diff = transforms.GaussianBlur(diff_blur_val, sigma=sigma_diff)(mask_diff.float())
            swapped_final_512_cxhxw_uint8 = (swapped_final_512_cxhxw_uint8.float() * mask_diff + original_face_512_cxhxw_uint8.float() * (1.0 - mask_diff)).byte()

        if parameters["AutoColorEnableToggle"]:
            ac_type = parameters['AutoColorTransferTypeSelection']
            ac_blend_param = parameters["AutoColorBlendAmountSlider"]

            # Pass the 512x512 mask for Test_Mask and DFL_Orig
            mask_for_autocolor_512 = t512(current_composite_mask_1x128x128)
            if ac_type == 'Test': swapped_final_512_cxhxw_uint8 = faceutil.histogram_matching(original_face_512_cxhxw_uint8, swapped_final_512_cxhxw_uint8, ac_blend_param)
            elif ac_type == 'Test_Mask': swapped_final_512_cxhxw_uint8 = faceutil.histogram_matching_withmask(original_face_512_cxhxw_uint8, swapped_final_512_cxhxw_uint8, mask_for_autocolor_512, ac_blend_param)
            elif ac_type == 'DFL_Test': swapped_final_512_cxhxw_uint8 = faceutil.histogram_matching_DFL_test(original_face_512_cxhxw_uint8, swapped_final_512_cxhxw_uint8, ac_blend_param)
            elif ac_type == 'DFL_Orig': swapped_final_512_cxhxw_uint8 = faceutil.histogram_matching_DFL_Orig(original_face_512_cxhxw_uint8, swapped_final_512_cxhxw_uint8, mask_for_autocolor_512, ac_blend_param)
            swapped_final_512_cxhxw_uint8 = swapped_final_512_cxhxw_uint8.byte()

        if parameters['ColorEnableToggle']:
            # Gamma needs to be applied carefully, often on float [0,1]
            temp_float_0_1 = swapped_final_512_cxhxw_uint8.float() / 255.0
            temp_float_0_1 = v2.functional.adjust_gamma(temp_float_0_1.unsqueeze(0), parameters['ColorGammaDecimalSlider'], 1.0).squeeze(0)
            swapped_final_512_cxhxw_uint8 = (temp_float_0_1 * 255.0).byte()

            temp_hwc_float = swapped_final_512_cxhxw_uint8.permute(1, 2, 0).float()
            del_color = torch.tensor([parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']], device=self.models_processor.device, dtype=torch.float32)
            temp_hwc_float += del_color
            temp_hwc_float = torch.clamp(temp_hwc_float, min=0., max=255.)
            swapped_final_512_cxhxw_uint8 = temp_hwc_float.permute(2,0,1).byte()

            swapped_final_512_cxhxw_uint8 = v2.functional.adjust_brightness(swapped_final_512_cxhxw_uint8, parameters['ColorBrightnessDecimalSlider'])
            swapped_final_512_cxhxw_uint8 = v2.functional.adjust_contrast(swapped_final_512_cxhxw_uint8, parameters['ColorContrastDecimalSlider'])
            swapped_final_512_cxhxw_uint8 = v2.functional.adjust_saturation(swapped_final_512_cxhxw_uint8, parameters['ColorSaturationDecimalSlider'])
            swapped_final_512_cxhxw_uint8 = v2.functional.adjust_sharpness(swapped_final_512_cxhxw_uint8, parameters['ColorSharpnessDecimalSlider'])
            swapped_final_512_cxhxw_uint8 = v2.functional.adjust_hue(swapped_final_512_cxhxw_uint8, parameters['ColorHueDecimalSlider']) 
            
            if parameters['ColorNoiseDecimalSlider'] > 0:

                temp_hwc_float_noise = swapped_final_512_cxhxw_uint8.permute(1, 2, 0).float()
                temp_hwc_float_noise += parameters['ColorNoiseDecimalSlider'] * torch.randn(512, 512, 3, device=self.models_processor.device)
                temp_hwc_float_noise = torch.clamp(temp_hwc_float_noise, 0, 255)
                swapped_final_512_cxhxw_uint8 = temp_hwc_float_noise.permute(2, 0, 1).byte()

        if parameters['JPEGCompressionEnableToggle']:
            try:
                swapped_final_512_cxhxw_uint8 = faceutil.jpegBlur(swapped_final_512_cxhxw_uint8, parameters["JPEGCompressionAmountSlider"])
            except Exception as e: print(f"JPEG Blur failed: {e}")

        final_blend_blur_val = parameters['FinalBlendAmountSlider'] 
        if parameters['FinalBlendAdjEnableToggle'] and final_blend_blur_val > 0:
            kernel_size = 2 * final_blend_blur_val + 1
            sigma = final_blend_blur_val * 0.1 + 1e-6 # Ensure sigma is positive
            if kernel_size > 1:
                swapped_final_512_cxhxw_uint8 = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(swapped_final_512_cxhxw_uint8)

        # Finalize the combined mask (including overall blur and border mask)
        # This needs to be done before checking is_perspective_crop, as the mask is returned in that case.
        overall_mask_blur_val = parameters['OverallMaskBlendAmountSlider'] * 2 + 1
        if overall_mask_blur_val > 1:
             sigma_overall = max(parameters['OverallMaskBlendAmountSlider']*0.15 + 0.1, 1e-6)
             current_composite_mask_1x128x128 = transforms.GaussianBlur(overall_mask_blur_val, sigma=sigma_overall)(current_composite_mask_1x128x128)

        # Combine the accumulated (and potentially blurred) mask with the border mask
        final_combined_mask_1x128x128 = torch.mul(current_composite_mask_1x128x128, border_mask_1x128x128)
        final_combined_mask_1x512x512 = t512(final_combined_mask_1x128x128) # Upscale to 512x512

        # If this is for a perspective crop, return the 512x512 swapped face directly
        if is_perspective_crop:
            # Also return the final combined mask for perspective crop pasting
            return swapped_final_512_cxhxw_uint8, final_combined_mask_1x512x512, None 
        
        # Prepare compare tensors if needed
        original_face_for_compare_tensor_hwc = None
        if self.is_view_face_compare:
            original_face_for_compare_tensor_hwc = original_face_512_cxhxw_uint8.permute(1, 2, 0) # HWC

        swap_mask_for_compare_tensor_hwc = None
        if self.is_view_face_mask:
            # Use the final combined mask for viewing
            mask_to_view_float_0_1 = final_combined_mask_1x512x512
            # Invert for "holes" view if desired, or use directly. Original code inverted.
            # mask_to_view_float_0_1 = 1.0 - mask_to_view_float_0_1 
            swap_mask_for_compare_tensor_hwc = (mask_to_view_float_0_1.repeat(3,1,1) * 255.0).byte().permute(1,2,0) # HWC

        # Apply the final combined mask to the swapped face before pasting
        final_combined_mask_3x512x512_float = final_combined_mask_1x512x512.repeat(3,1,1).float()
        masked_swapped_face_to_paste_float = swapped_final_512_cxhxw_uint8.float() * final_combined_mask_3x512x512_float

        # Get grid for pasting onto the full original image
        img_h, img_w = img_cxhxw_rgb_uint8.shape[1], img_cxhxw_rgb_uint8.shape[2]
        # tform maps from img_cxhxw_rgb_uint8 to 512 template. We need this for grid_sample.
        _, source_grid_normalized_xy_for_paste = self.get_grid_for_pasting(
            tform, img_h, img_w, 512, 512, img_cxhxw_rgb_uint8.device
        )
        
        # Warp the masked swapped face to the full image canvas
        pasted_face_on_full_image_float = torch.nn.functional.grid_sample(
            masked_swapped_face_to_paste_float.unsqueeze(0),
            source_grid_normalized_xy_for_paste, # Grid maps target (full_img) to source (512_face)
            mode='bilinear',
            padding_mode='border', # Use border for image data
            align_corners=False
        ).squeeze(0)

        # Warp the mask to the full image canvas
        transformed_mask_on_canvas_float = torch.nn.functional.grid_sample(
            final_combined_mask_3x512x512_float.unsqueeze(0), # Use the same mask
            source_grid_normalized_xy_for_paste,
            mode='bilinear', padding_mode='zeros', # Mask should be zero outside
            align_corners=False
        ).squeeze(0)
        
        # Blend with the original full image
        original_img_float = img_cxhxw_rgb_uint8.float()
        blended_full_image_float = pasted_face_on_full_image_float + \
                                   original_img_float * (1.0 - transformed_mask_on_canvas_float)
        
        img_cxhxw_rgb_uint8_output = torch.clamp(blended_full_image_float, 0, 255).byte()
        
        return img_cxhxw_rgb_uint8_output, original_face_for_compare_tensor_hwc, swap_mask_for_compare_tensor_hwc

    def get_grid_for_pasting(self, tform_target_to_source: trans.SimilarityTransform,
                             target_h: int, target_w: int,
                             source_h: int, source_w: int,
                             device: torch.device):
        # tform_target_to_source: maps points from target (e.g. full image) to source (e.g. 512x512 face)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(target_h, device=device, dtype=torch.float32),
            torch.arange(target_w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        target_grid_yx_pixels = torch.stack((grid_y, grid_x), dim=2).unsqueeze(0) # 1xTargetHxTargetWx2 (Y,X order)

        # Convert target grid pixel coordinates to homogeneous coordinates (X,Y,1)
        target_grid_xy_flat_pixels = target_grid_yx_pixels[..., [1,0]].reshape(-1, 2) # (N,2) in XY
        ones = torch.ones(target_grid_xy_flat_pixels.shape[0], 1, device=device, dtype=torch.float32)
        homogeneous_target_grid_xy_pixels = torch.cat((target_grid_xy_flat_pixels, ones), dim=1) # (N,3)

        # Transformation matrix from tform_target_to_source (2x3)
        M_target_to_source = torch.tensor(tform_target_to_source.params[0:2, :], dtype=torch.float32, device=device)

        # Transform target grid to source coordinates (pixels)
        # (N,3) @ (3,2) -> (N,2) in XY order
        source_coords_xy_flat_pixels = torch.matmul(homogeneous_target_grid_xy_pixels, M_target_to_source.T)

        # Reshape to grid format 1xTargetHxTargetWx2
        source_coords_xy_grid_pixels = source_coords_xy_flat_pixels.view(1, target_h, target_w, 2)

        # Normalize source coordinates for grid_sample (expects XY order, range [-1,1])
        source_grid_normalized_xy = torch.empty_like(source_coords_xy_grid_pixels)
        # Normalize X coordinates
        source_grid_normalized_xy[..., 0] = (source_coords_xy_grid_pixels[..., 0] / (source_w - 1.0)) * 2.0 - 1.0
        # Normalize Y coordinates
        source_grid_normalized_xy[..., 1] = (source_coords_xy_grid_pixels[..., 1] / (source_h - 1.0)) * 2.0 - 1.0
        
        # target_grid_yx is not strictly needed by grid_sample but returned for completeness if ever useful
        return target_grid_yx_pixels, source_grid_normalized_xy

    def enhance_core(self, img_cxhxw_rgb_uint8: torch.Tensor, control:dict) -> torch.Tensor:
        # img_cxhxw_rgb_uint8 is CxHxW, Torch tensor on GPU, uint8
        enhancer_type = control.get('FrameEnhancerTypeSelection', None)
        output_img_tensor = img_cxhxw_rgb_uint8.clone() # Start with a clone

        if not enhancer_type: return output_img_tensor # No enhancer selected

        match enhancer_type:
            case 'RealEsrgan-x2-Plus' | 'RealEsrgan-x4-Plus' | 'BSRGan-x2' | 'BSRGan-x4' | \
                 'UltraSharp-x4' | 'UltraMix-x4' | 'RealEsr-General-x4v3':
                tile_size = 512
                scale = 2 if enhancer_type in ('RealEsrgan-x2-Plus', 'BSRGan-x2') else 4
                
                image_float_0_1 = img_cxhxw_rgb_uint8.float() / 255.0 # Normalize to [0,1]
                image_batch_float = image_float_0_1.unsqueeze(0) # Add batch dimension

                enhanced_batch_float = self.models_processor.run_enhance_frame_tile_process(
                    image_batch_float, enhancer_type, tile_size=tile_size, scale=scale
                )
                
                enhanced_cxhxw_float_0_1 = torch.clamp(enhanced_batch_float.squeeze(0), 0, 1)
                
                alpha = float(control.get("FrameEnhancerBlendSlider", 100))/100.0
                
                # Resize original image to match enhanced image dimensions for blending
                resized_original_for_blend = v2.Resize(
                    (enhanced_cxhxw_float_0_1.shape[1], enhanced_cxhxw_float_0_1.shape[2]),
                    interpolation=v2.InterpolationMode.BILINEAR, antialias=True
                )(image_float_0_1)
                
                blended_float_0_1 = enhanced_cxhxw_float_0_1 * alpha + resized_original_for_blend * (1.0 - alpha)
                output_img_tensor = (torch.clamp(blended_float_0_1 * 255.0, 0, 255)).byte()

            case 'DeOldify-Artistic' | 'DeOldify-Stable' | 'DeOldify-Video':
                render_factor = 384 # Fixed render factor for DeOldify
                _, h, w = img_cxhxw_rgb_uint8.shape
                
                # Resize input for DeOldify model
                resized_for_deoldify_uint8 = v2.Resize(
                    (render_factor, render_factor),
                    interpolation=v2.InterpolationMode.BILINEAR, antialias=False
                )(img_cxhxw_rgb_uint8)
                
                image_float_0_255_for_deoldify = resized_for_deoldify_uint8.float() # DeOldify might expect 0-255 float
                image_batch_float_for_deoldify = image_float_0_255_for_deoldify.unsqueeze(0)

                deoldify_output_batch_float = torch.empty_like(image_batch_float_for_deoldify)

                if enhancer_type == 'DeOldify-Artistic':
                    self.models_processor.run_deoldify_artistic(image_batch_float_for_deoldify, deoldify_output_batch_float)
                elif enhancer_type == 'DeOldify-Stable':
                    self.models_processor.run_deoldify_stable(image_batch_float_for_deoldify, deoldify_output_batch_float)
                else: # 'DeOldify-Video'
                    self.models_processor.run_deoldify_video(image_batch_float_for_deoldify, deoldify_output_batch_float)

                deoldify_output_cxhxw_float_0_255 = deoldify_output_batch_float.squeeze(0)
                # Resize DeOldify output back to original frame dimensions
                resized_deoldify_output_0_255 = v2.Resize(
                    (h, w), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
                )(deoldify_output_cxhxw_float_0_255)

                # Convert to [0,1] float for YUV processing
                img_float_0_1 = img_cxhxw_rgb_uint8.float() / 255.0
                deoldify_output_float_0_1 = torch.clamp(resized_deoldify_output_0_255 / 255.0, 0, 1)

                output_yuv = faceutil.rgb_to_yuv(deoldify_output_float_0_1, normalize=False) # Assuming faceutil handles [0,1]
                hires_yuv = faceutil.rgb_to_yuv(img_float_0_1, normalize=False)

                hires_yuv[1:3, :, :] = output_yuv[1:3, :, :] # Replace Cb, Cr channels
                hires_rgb_float_0_1 = faceutil.yuv_to_rgb(hires_yuv, normalize=False)
                
                alpha = float(control.get("FrameEnhancerBlendSlider", 100)) / 100.0
                blended_float_0_1 = hires_rgb_float_0_1 * alpha + img_float_0_1 * (1.0 - alpha)
                output_img_tensor = (torch.clamp(blended_float_0_1 * 255.0, 0, 255)).byte()

            case 'DDColor-Artistic' | 'DDColor':
                render_factor = 384
                # Original L channel from the input image (normalized to [0,100] for L, [-128,127] for ab)
                orig_lab_normalized = faceutil.rgb_to_lab(img_cxhxw_rgb_uint8, normalize=True)
                orig_l_channel_normalized = orig_lab_normalized[0:1, :, :] # L channel

                # Resize input image for DDColor model
                resized_for_ddcolor_uint8 = v2.Resize(
                    (render_factor, render_factor),
                    interpolation=v2.InterpolationMode.BILINEAR, antialias=False
                )(img_cxhxw_rgb_uint8)
                
                # Convert resized to LAB and get L channel for DDColor input (as grayscale RGB)
                lab_for_ddcolor_input_normalized = faceutil.rgb_to_lab(resized_for_ddcolor_uint8, normalize=True)
                l_channel_for_ddcolor_input_normalized = lab_for_ddcolor_input_normalized[0:1,:,:]
                
                # Create grayscale LAB image (L, 0, 0) then convert to RGB for model input
                gray_lab_for_ddcolor_normalized = torch.cat((
                    l_channel_for_ddcolor_input_normalized,
                    torch.zeros_like(l_channel_for_ddcolor_input_normalized), # a channel = 0
                    torch.zeros_like(l_channel_for_ddcolor_input_normalized)  # b channel = 0
                ), dim=0)
                gray_rgb_for_ddcolor_float_0_1 = faceutil.lab_to_rgb(gray_lab_for_ddcolor_normalized, normalize=True)
                
                tensor_gray_rgb_batch_float_0_1 = gray_rgb_for_ddcolor_float_0_1.float().unsqueeze(0)
                
                # DDColor predicts ab channels (normalized)
                predicted_ab_batch_normalized = torch.empty((1, 2, render_factor, render_factor), dtype=torch.float32, device=self.models_processor.device)
                if enhancer_type == 'DDColor-Artistic':
                    self.models_processor.run_ddcolor_artistic(tensor_gray_rgb_batch_float_0_1, predicted_ab_batch_normalized)
                else: # 'DDColor'
                    self.models_processor.run_ddcolor(tensor_gray_rgb_batch_float_0_1, predicted_ab_batch_normalized)
                
                predicted_ab_cxhxw_normalized = predicted_ab_batch_normalized.squeeze(0)

                # Resize predicted ab channels to original image size
                resized_predicted_ab_normalized = v2.Resize(
                    (img_cxhxw_rgb_uint8.shape[1], img_cxhxw_rgb_uint8.shape[2]),
                    interpolation=v2.InterpolationMode.BILINEAR, antialias=False
                )(predicted_ab_cxhxw_normalized)
                
                # Combine original L channel with predicted ab channels
                final_lab_output_normalized = torch.cat((orig_l_channel_normalized, resized_predicted_ab_normalized), dim=0)
                colorized_rgb_float_0_1 = faceutil.lab_to_rgb(final_lab_output_normalized, normalize=True)
                                                                                             
                alpha = float(control.get("FrameEnhancerBlendSlider", 100)) / 100.0
                img_float_0_1 = img_cxhxw_rgb_uint8.float() / 255.0 # Original image [0,1]
                blended_float_0_1 = colorized_rgb_float_0_1 * alpha + img_float_0_1 * (1.0 - alpha)
                output_img_tensor = (torch.clamp(blended_float_0_1 * 255.0, 0, 255)).byte()
                
        return output_img_tensor

    def apply_face_expression_restorer(self, driving_cxhxw_uint8: torch.Tensor, target_cxhxw_uint8: torch.Tensor, parameters: dict) -> torch.Tensor:
        # driving_cxhxw_uint8, target_cxhxw_uint8 are Cx512x512 uint8 RGB
        t256_resize = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        # Process driving face
        # Assuming driving_cxhxw_uint8 is already aligned 512x512 face
        _, driving_lmk_crop_list, _ = self.models_processor.run_detect_landmark(
            driving_cxhxw_uint8, bbox=np.array([0, 0, 512, 512]), det_kpss=[],
            detect_mode='203', score=0.5, from_points=False
        )
        if not driving_lmk_crop_list: return target_cxhxw_uint8 # Cannot proceed
        driving_lmk_crop = driving_lmk_crop_list[0]

        driving_face_256 = t256_resize(driving_cxhxw_uint8)
        c_d_eyes_lst = faceutil.calc_eye_close_ratio(driving_lmk_crop[None])
        c_d_lip_lst = faceutil.calc_lip_close_ratio(driving_lmk_crop[None])
        x_d_i_info = self.models_processor.lp_motion_extractor(driving_face_256, 'Human-Face') # Assuming 'Human-Face' is default
        R_d_i = faceutil.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
                
        # Get parameters from UI (direct access)
        driving_multiplier = parameters['FaceExpressionFriendlyFactorDecimalSlider']
        animation_region_str = parameters['FaceExpressionAnimationRegionSelection']
        flag_normalize_lip = parameters['FaceExpressionNormalizeLipsEnableToggle']
        lip_normalize_threshold = parameters['FaceExpressionNormalizeLipsThresholdDecimalSlider']
        flag_eye_retargeting = parameters['FaceExpressionRetargetingEyesEnableToggle']
        eye_retargeting_multiplier = parameters['FaceExpressionRetargetingEyesMultiplierDecimalSlider']
        flag_lip_retargeting = parameters['FaceExpressionRetargetingLipsEnableToggle']
        lip_retargeting_multiplier = parameters['FaceExpressionRetargetingLipsMultiplierDecimalSlider']
        
        flag_relative_motion = True # Default from original logic
        flag_stitching = True       # Default
        flag_pasteback = True       # Default
        flag_do_crop = True         # Default
        
        # Process target face (which is the swapped face)
        # Assuming target_cxhxw_uint8 is also an aligned 512x512 face
        _, source_lmk_list, _ = self.models_processor.run_detect_landmark(
            target_cxhxw_uint8, bbox=np.array([0, 0, 512, 512]), det_kpss=[],
            detect_mode='203', score=0.5, from_points=False
        )
        if not source_lmk_list: return target_cxhxw_uint8 # Cannot proceed
        source_lmk = source_lmk_list[0]

        # Warp target face based on its own landmarks for consistent processing space
        target_face_512_warped, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
            target_cxhxw_uint8, source_lmk, dsize=512,
            scale=parameters['FaceExpressionCropScaleDecimalSlider'],
            vy_ratio=parameters['FaceExpressionVYRatioDecimalSlider'],
            interpolation=v2.InterpolationMode.BILINEAR
        )
        target_face_256_warped = t256_resize(target_face_512_warped)

        x_s_info = self.models_processor.lp_motion_extractor(target_face_256_warped, 'Human-Face')
        x_c_s = x_s_info['kp'] # Canonical keypoints of source
        R_s = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.models_processor.lp_appearance_feature_extractor(target_face_256_warped, 'Human-Face')
        x_s_transformed_kp = faceutil.transform_keypoint(x_s_info) # Transformed keypoints of source

        # Lip normalization adjustment (if enabled)
        lip_delta_before_animation = None
        if flag_normalize_lip and flag_relative_motion and source_lmk is not None:
            # Use a neutral lip state for 'before animation'
            c_d_lip_before_animation_neutral = [0.] # Represents closed or neutral lip
            combined_lip_ratio_tensor_before_animation = faceutil.calc_combined_lip_ratio(
                c_d_lip_before_animation_neutral, source_lmk, device=self.models_processor.device
            )
            if combined_lip_ratio_tensor_before_animation[0][0] >= lip_normalize_threshold:
                lip_delta_before_animation = self.models_processor.lp_retarget_lip(
                    x_s_transformed_kp, combined_lip_ratio_tensor_before_animation
                )

        # Calculate new expression and pose based on driving face and animation region
        delta_new_exp = x_s_info['exp'].clone() # Start with source expression
        R_d_0_pose = R_d_i.clone() # Initial driving pose
        x_d_0_info_exp = x_d_i_info.copy() # Initial driving expression info

        if flag_relative_motion:
            if animation_region_str == "all" or "pose" in animation_region_str:
                R_new_pose = (R_d_i @ R_d_0_pose.permute(0, 2, 1)) @ R_s
            else: R_new_pose = R_s.clone()
            
            if animation_region_str == "all" or "exp" in animation_region_str:
                # Apply full relative expression change
                delta_new_exp = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info_exp['exp'])
            else: # Selective expression change
                if "lips" in animation_region_str:
                    for lip_idx in [6, 12, 14, 17, 19, 20]: # Example lip indices
                        delta_new_exp[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info_exp['exp']))[:, lip_idx, :]
                if "eyes" in animation_region_str:
                     for eyes_idx in [11, 13, 15, 16, 18]: # Example eye indices
                        delta_new_exp[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info_exp['exp']))[:, eyes_idx, :]
            
            if animation_region_str == "all" or "pose" in animation_region_str:
                scale_new_pose = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info_exp['scale'])
                t_new_translation = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info_exp['t'])
            else:
                scale_new_pose = x_s_info['scale'].clone()
                t_new_translation = x_s_info['t'].clone()
        else: # Absolute motion
             if animation_region_str == "all" or "pose" in animation_region_str: R_new_pose = R_d_i.clone()
             else: R_new_pose = R_s.clone()
             
             if animation_region_str == "all" or "exp" in animation_region_str:
                 delta_new_exp = x_d_i_info['exp'].clone()
             else:
                 if "lips" in animation_region_str:
                    for lip_idx in [6, 12, 14, 17, 19, 20]: delta_new_exp[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :].clone()
                 if "eyes" in animation_region_str:
                    for eyes_idx in [11, 13, 15, 16, 18]: delta_new_exp[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :].clone()

             scale_new_pose = x_s_info['scale'].clone()
             if animation_region_str == "all" or "pose" in animation_region_str: t_new_translation = x_d_i_info['t'].clone()
             else: t_new_translation = x_s_info['t'].clone()

        t_new_translation[..., 2].fill_(0) # Zero out z-translation
        # Calculate new transformed keypoints (x_d_i_new)
        x_d_i_new_transformed_kp = scale_new_pose * (x_c_s @ R_new_pose + delta_new_exp) + t_new_translation
        
        # Apply lip normalization delta if calculated and conditions met
        if flag_normalize_lip and lip_delta_before_animation is not None and \
           not (flag_stitching or flag_eye_retargeting or flag_lip_retargeting): # Only if no other retargeting
             x_d_i_new_transformed_kp += lip_delta_before_animation

        # Eye and Lip Retargeting
        eyes_delta_retarget, lip_delta_retarget = None, None
        if flag_eye_retargeting and source_lmk is not None:
            combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio(c_d_eyes_lst, source_lmk, device=self.models_processor.device)
            combined_eye_ratio_tensor = combined_eye_ratio_tensor * eye_retargeting_multiplier
            eyes_delta_retarget = self.models_processor.lp_retarget_eye(x_s_transformed_kp, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])

        if flag_lip_retargeting and source_lmk is not None:
            combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio(c_d_lip_lst, source_lmk, device=self.models_processor.device)
            combined_lip_ratio_tensor = combined_lip_ratio_tensor * lip_retargeting_multiplier
            lip_delta_retarget = self.models_processor.lp_retarget_lip(x_s_transformed_kp, combined_lip_ratio_tensor, parameters["FaceEditorTypeSelection"])
        
        # Accumulate retargeting deltas
        final_retargeting_delta = torch.zeros_like(x_s_transformed_kp) # Ensure correct shape
        if eyes_delta_retarget is not None: final_retargeting_delta += eyes_delta_retarget
        if lip_delta_retarget is not None: final_retargeting_delta += lip_delta_retarget
        
        # Apply retargeting delta based on motion mode
        if flag_relative_motion:
             # Add retargeting delta to the already relative motion adjusted keypoints
             x_d_i_new_transformed_kp = x_s_transformed_kp + (x_d_i_new_transformed_kp - x_s_transformed_kp) + final_retargeting_delta
        else: # Absolute motion
             x_d_i_new_transformed_kp = x_d_i_new_transformed_kp + final_retargeting_delta

        # Stitching
        if flag_stitching:
            x_d_i_new_transformed_kp = self.models_processor.lp_stitching(
                x_s_transformed_kp, x_d_i_new_transformed_kp, parameters["FaceEditorTypeSelection"]
            )
            # Apply lip normalization delta after stitching if it was deferred
            if lip_delta_before_animation is not None and (flag_stitching or flag_eye_retargeting or flag_lip_retargeting):
                x_d_i_new_transformed_kp += lip_delta_before_animation

        # Apply driving multiplier
        x_d_i_new_transformed_kp = x_s_transformed_kp + (x_d_i_new_transformed_kp - x_s_transformed_kp) * driving_multiplier
        
        # Decode (generate new face)
        out_float_0_1 = self.models_processor.lp_warp_decode(
            f_s, x_s_transformed_kp, x_d_i_new_transformed_kp, parameters["FaceEditorTypeSelection"]
        )
        out_float_0_1 = torch.squeeze(out_float_0_1)
        out_float_0_1 = torch.clamp(out_float_0_1, 0, 1)

        # Paste back onto the original target_cxhxw_uint8 (which was the input swapped face)
        if flag_pasteback and flag_do_crop:
            target_float_0_1 = target_cxhxw_uint8.float() / 255.0 # Original swapped face [0,1]
            # The output of warp_decode is 256x256, needs to be 512x512 for paste_back with M_c2o from 512 warp
            out_resized_to_512_float_0_1 = t512(out_float_0_1)
            
            # Mask for pasting (lp_mask_crop is 256x256, need 512x512)
            mask_for_pasteback_512 = t512(self.models_processor.lp_mask_crop)

            output_pasted_float_0_1 = faceutil.paste_back_adv( # Use paste_back_adv
                out_resized_to_512_float_0_1,
                M_c2o, # Transform from warped 512 space back to original 512 space
                target_float_0_1, # Paste onto the original swapped face
                mask_for_pasteback_512 # Use the upscaled mask
            )
            final_output_uint8 = (torch.clamp(output_pasted_float_0_1 * 255.0, 0, 255)).byte()
        else: # No paste back, just return the generated face (resized to 512)
            final_output_uint8 = (torch.clamp(t512(out_float_0_1) * 255.0, 0, 255)).byte()
            
        return final_output_uint8

    def swap_edit_face_core(self, img_cxhxw_rgb_uint8: torch.Tensor, kps_all: np.ndarray | None,
                            parameters: dict, control: dict, **kwargs) -> torch.Tensor:
        # img_cxhxw_rgb_uint8: Full frame or perspective crop, CxHxW uint8 RGB
        # kps_all: All keypoints for the face in img_cxhxw_rgb_uint8, or None
        
        img_output_tensor = img_cxhxw_rgb_uint8.clone() # Work on a clone

        if parameters['FaceEditorEnableToggle']:
            t256_resize = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
            
            # --- Landmark Detection ---
            lmk_crop_for_edit = None
            if kps_all is not None and kps_all.size > 0:
                 # Try to get landmarks from provided kps_all
                 _, lmk_crop_list, _ = self.models_processor.run_detect_landmark(
                     img_output_tensor, bbox=[], det_kpss=kps_all,
                     detect_mode='203', score=0.5, from_points=True
                 )
                 if lmk_crop_list: lmk_crop_for_edit = lmk_crop_list[0] # Assuming first face if multiple from kps_all

            if lmk_crop_for_edit is None: # Fallback to full detection if kps_all failed or not provided
                # This detection is on the potentially large img_output_tensor
                temp_bboxes, _, temp_lmk_crop_list = self.models_processor.run_detect(
                    img_output_tensor, control['DetectorModelSelection'], max_num=1, score=0.5,
                    input_size=(img_output_tensor.shape[1], img_output_tensor.shape[2]), # H, W
                    use_landmark_detection=True, landmark_detect_mode='203'
                )
                if temp_lmk_crop_list and len(temp_lmk_crop_list) > 0:
                    lmk_crop_for_edit = temp_lmk_crop_list[0]
                else: # Still no landmarks, cannot proceed with editor
                    return img_cxhxw_rgb_uint8 # Return original

            # --- Face Warping and Feature Extraction ---
            # Warp the face from the full image to a 512x512 aligned template
            original_face_512_warped, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
                img_output_tensor, lmk_crop_for_edit, dsize=512,
                scale=parameters['FaceEditorCropScaleDecimalSlider'],
                vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'],
                interpolation=v2.InterpolationMode.BILINEAR
            )
            original_face_256_warped = t256_resize(original_face_512_warped)

            x_s_info = self.models_processor.lp_motion_extractor(original_face_256_warped, parameters["FaceEditorTypeSelection"])
            R_s_user = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            f_s_user = self.models_processor.lp_appearance_feature_extractor(original_face_256_warped, parameters["FaceEditorTypeSelection"])
            x_s_transformed_kp = faceutil.transform_keypoint(x_s_info) # Source keypoints in canonical space

            # --- Apply User Edits (Pose, Expression, Movement) ---
            # Head Pose
            x_d_info_user_pitch = x_s_info['pitch'] + parameters['HeadPitchSlider']
            x_d_info_user_yaw = x_s_info['yaw'] + parameters['HeadYawSlider']
            x_d_info_user_roll = x_s_info['roll'] + parameters['HeadRollSlider']
            R_d_user_edited_pose = faceutil.get_rotation_matrix(x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll)
            
            # Expression Deltas
            delta_new_exp = x_s_info['exp'].clone() # Start with source expression

            if parameters['EyeGazeHorizontalDecimalSlider'] != 0 or parameters['EyeGazeVerticalDecimalSlider'] != 0:
                 delta_new_exp = faceutil.update_delta_new_eyeball_direction(parameters['EyeGazeHorizontalDecimalSlider'], parameters['EyeGazeVerticalDecimalSlider'], delta_new_exp)
            if parameters['MouthSmileDecimalSlider'] != 0: delta_new_exp = faceutil.update_delta_new_smile(parameters['MouthSmileDecimalSlider'], delta_new_exp)

            # ... (add other expression sliders similarly using .get with defaults) ...
            if parameters['XAxisMovementDecimalSlider'] != 0: delta_new_exp = faceutil.update_delta_new_mov_x(-parameters['XAxisMovementDecimalSlider'], delta_new_exp)
            if parameters['YAxisMovementDecimalSlider'] != 0: delta_new_exp = faceutil.update_delta_new_mov_y(parameters['YAxisMovementDecimalSlider'], delta_new_exp)

            # Combine pose and expression
            x_c_s_canonical_kp = x_s_info['kp'] # Canonical keypoints
            scale_new_edited = x_s_info['scale'] * parameters['ZAxisMovementDecimalSlider'] # Z-axis as scale
            t_new_edited_translation = x_s_info['t'].clone() # Start with source translation
            
            # Final target pose (combining user edit with original source pose)
            R_d_final_pose = (R_d_user_edited_pose @ R_s_user.permute(0, 2, 1)) @ R_s_user
            
            # New transformed keypoints with edits
            x_d_new_edited_transformed_kp = scale_new_edited * (x_c_s_canonical_kp @ R_d_final_pose + delta_new_exp) + t_new_edited_translation

            # --- Eye/Lip Retargeting (based on sliders relative to detected state) ---
            eyes_delta_retarget, lip_delta_retarget = None, None
            source_eye_ratio_calc = faceutil.calc_eye_close_ratio(lmk_crop_for_edit[None]) # Needs batch dim
            init_source_eye_ratio = round(float(source_eye_ratio_calc.mean()), 2)
            target_eye_open_ratio = max(min(init_source_eye_ratio + parameters['EyesOpenRatioDecimalSlider'], 0.80), 0.00)
            if abs(target_eye_open_ratio - init_source_eye_ratio) > 1e-3:
                combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio([[target_eye_open_ratio]], lmk_crop_for_edit, device=self.models_processor.device)
                eyes_delta_retarget = self.models_processor.lp_retarget_eye(x_s_transformed_kp, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])

            source_lip_ratio_calc = faceutil.calc_lip_close_ratio(lmk_crop_for_edit[None]) # Needs batch dim
            init_source_lip_ratio = round(float(source_lip_ratio_calc[0][0]), 2)
            target_lip_open_ratio = max(min(init_source_lip_ratio + parameters['LipsOpenRatioDecimalSlider'], 0.80), 0.00)

            if abs(target_lip_open_ratio - init_source_lip_ratio) > 1e-3:
                combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio([[target_lip_open_ratio]], lmk_crop_for_edit, device=self.models_processor.device)
                lip_delta_retarget = self.models_processor.lp_retarget_lip(x_s_transformed_kp, combined_lip_ratio_tensor, parameters["FaceEditorTypeSelection"])

            if eyes_delta_retarget is not None: x_d_new_edited_transformed_kp = x_d_new_edited_transformed_kp + eyes_delta_retarget
            if lip_delta_retarget is not None: x_d_new_edited_transformed_kp = x_d_new_edited_transformed_kp + lip_delta_retarget

            # Stitching
            if kwargs.get('flag_stitching_retargeting_input', True): # Default to True
                x_d_new_edited_transformed_kp = self.models_processor.lp_stitching(
                    x_s_transformed_kp, x_d_new_edited_transformed_kp, parameters["FaceEditorTypeSelection"]
                )
            
            # Decode to get the edited face image (256x256 float [0,1])
            out_edited_face_256_float_0_1 = self.models_processor.lp_warp_decode(
                f_s_user, x_s_transformed_kp, x_d_new_edited_transformed_kp, parameters["FaceEditorTypeSelection"]
            )
            out_edited_face_256_float_0_1 = torch.squeeze(out_edited_face_256_float_0_1)
            out_edited_face_256_float_0_1 = torch.clamp(out_edited_face_256_float_0_1, 0, 1)

            # --- Paste back the edited face ---
            if kwargs.get('flag_do_crop_input_retargeting_image', True): # Default to True
                blur_kernel_size = parameters['FaceEditorBlurAmountSlider']*2+1
                mask_blur_sigma = max(parameters['FaceEditorBlurAmountSlider']*0.15 + 0.1, 1e-6)
                
                # lp_mask_crop is 1x256x256, ensure it's used correctly
                mask_crop_for_paste = self.models_processor.lp_mask_crop.clone()
                if blur_kernel_size > 1:
                     mask_crop_for_paste = transforms.GaussianBlur(blur_kernel_size, mask_blur_sigma)(mask_crop_for_paste)
                
                img_output_float_0_1 = img_output_tensor.float() / 255.0
                # Edited face is 256x256, M_c2o is for 512x512. We need to paste 256 onto full image.
                # This requires adjusting M_c2o or pasting the 512 warped version.
                # For simplicity, let's assume paste_back_adv handles the 256x256 source with 512x512 M_c2o
                # by effectively operating on a 512x512 canvas where the 256x256 is centered.
                # Or, more correctly, warp_decode output (256) should be resized to 512 before paste_back_adv if M_c2o is for 512.
                out_edited_face_512_float_0_1 = t512(out_edited_face_256_float_0_1)
                mask_for_paste_512 = t512(mask_crop_for_paste)


                img_output_float_0_1 = faceutil.paste_back_adv(
                    out_edited_face_512_float_0_1, M_c2o, img_output_float_0_1, mask_for_paste_512
                )
                img_output_tensor = (torch.clamp(img_output_float_0_1 * 255.0, 0, 255)).byte()
            else: # No paste back, just return the 512x512 edited face
                img_output_tensor = (torch.clamp(t512(out_edited_face_256_float_0_1) * 255.0, 0, 255)).byte()

        # --- Makeup Application (Applied after editor, on the potentially edited full image) ---
        if parameters['FaceMakeupEnableToggle'] or parameters['HairMakeupEnableToggle'] or \
           parameters['EyeBrowsMakeupEnableToggle'] or parameters['LipsMakeupEnableToggle']:
            
            # Re-detect landmarks on the current state of img_output_tensor for makeup alignment
            lmk_crop_for_makeup = None
            # Similar fallback detection as above if kps_all is not good for makeup
            if kps_all is not None and kps_all.size > 0:
                 _, lmk_crop_list_mu, _ = self.models_processor.run_detect_landmark(img_output_tensor, bbox=[], det_kpss=kps_all, detect_mode='203', score=0.5, from_points=True)
                 if lmk_crop_list_mu: lmk_crop_for_makeup = lmk_crop_list_mu[0]

            if lmk_crop_for_makeup is None:
                temp_bboxes_mu, _, temp_lmk_crop_list_mu = self.models_processor.run_detect(img_output_tensor, control['DetectorModelSelection'], max_num=1, score=0.5, input_size=(img_output_tensor.shape[1], img_output_tensor.shape[2]), use_landmark_detection=True, landmark_detect_mode='203')
                if temp_lmk_crop_list_mu and len(temp_lmk_crop_list_mu) > 0:
                    lmk_crop_for_makeup = temp_lmk_crop_list_mu[0]
                else: # Cannot apply makeup without landmarks
                    return img_output_tensor

            # Warp face for makeup
            face_512_for_makeup_warped, M_o2c_mu, M_c2o_mu = faceutil.warp_face_by_face_landmark_x(
                img_output_tensor, lmk_crop_for_makeup, dsize=512,
                scale=parameters['FaceEditorCropScaleDecimalSlider'], # Reuse editor's crop scale
                vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'],
                interpolation=v2.InterpolationMode.BILINEAR
            )

            makeup_out_512_uint8, makeup_mask_out_512_0_1_float = self.models_processor.apply_face_makeup(face_512_for_makeup_warped, parameters)
            
            img_output_float_0_1 = img_output_tensor.float() / 255.0
            makeup_out_512_float_0_1 = makeup_out_512_uint8.float() / 255.0
            
            # Ensure makeup_mask_out_512_0_1_float is 1xHxW for paste_back_adv
            if makeup_mask_out_512_0_1_float.ndim == 2: # HxW
                makeup_mask_out_512_0_1_float = makeup_mask_out_512_0_1_float.unsqueeze(0) # 1xHxW
            
            img_output_float_0_1 = faceutil.paste_back_adv(
                makeup_out_512_float_0_1, M_c2o_mu, img_output_float_0_1, makeup_mask_out_512_0_1_float
            )
            img_output_tensor = (torch.clamp(img_output_float_0_1 * 255.0, 0, 255)).byte()

        return img_output_tensor