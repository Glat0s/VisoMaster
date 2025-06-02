import threading
import os
import subprocess as sp
import gc
import traceback
from typing import Dict, TYPE_CHECKING

from packaging import version
import numpy as np
import onnxruntime
import torch
import onnx
from torchvision.transforms import v2
from PySide6 import QtCore
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ModuleNotFoundError:
    print("No TensorRT Found")
    TENSORRT_AVAILABLE = False

from app.processors.utils.engine_builder import onnx_to_trt as onnx2trt
from app.processors.utils.tensorrt_predictor import TensorRTPredictor
from app.processors.face_detectors import FaceDetectors
from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.face_masks import FaceMasks
from app.processors.face_restorers import FaceRestorers
from app.processors.face_swappers import FaceSwappers
from app.processors.frame_enhancers import FrameEnhancers
from app.processors.face_editors import FaceEditors
from app.processors.utils.dfm_model import DFMModel
from app.processors.models_data import models_list, arcface_mapping_model_dict, models_trt_list, models_dir
from app.processors.utils import faceutil
from app.helpers.miscellaneous import is_file_exists
from app.helpers.downloader import download_file

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1
lock = threading.Lock()

class ModelsProcessor(QtCore.QObject):
    processing_complete = QtCore.Signal()
    model_loaded = QtCore.Signal()  # Signal emitted with Onnx InferenceSession

    def __init__(self, main_window: 'MainWindow', device='cuda'):
        super().__init__()
        self.main_window = main_window
        self.provider_name = 'TensorRT'
        # Initialize internal cache for K/V tensors
        self.internal_deep_copied_kv_map: Dict[str, Dict[str, torch.Tensor]] | None = None
        self.internal_kv_map_source_filename: str | None = None

        self.device = device
        self.model_lock = threading.RLock()  # Reentrant lock for model access
        # TensorRT Execution Provider options
        self.trt_ep_options = {
            # 'trt_max_workspace_size': 3 << 30,  # Dimensione massima dello spazio di lavoro in bytes
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': "tensorrt-engines",
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': "tensorrt-engines",
            'trt_dump_ep_context_model': True,
            'trt_ep_context_file_path': "tensorrt-engines",
            'trt_layer_norm_fp32_fallback': True,
            'trt_builder_optimization_level': 5,
        }
        self.providers = [
            ('CUDAExecutionProvider'),
            ('CPUExecutionProvider')
        ]       
        self.nThreads = 2
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device=self.device)

        # Initialize models and models_path
        self.models: Dict[str, onnxruntime.InferenceSession] = {}
        self.models_path = {}
        self.models_data = {}
        for model_data in models_list:
            model_name, model_path = model_data['model_name'], model_data['local_path']
            self.models[model_name] = None #Model Instance
            self.models_path[model_name] = model_path
            self.models_data[model_name] = {'local_path': model_data['local_path'], 'hash': model_data['hash'], 'url': model_data.get('url')}

        self.dfm_models: Dict[str, DFMModel] = {}

        if TENSORRT_AVAILABLE:
            # Initialize models_trt and models_trt_path
            self.models_trt = {}
            self.models_trt_path = {}
            for model_data in models_trt_list:
                model_name, model_path = model_data['model_name'], model_data['local_path']
                self.models_trt[model_name] = None #Model Instance
                self.models_trt_path[model_name] = model_path

        self.face_detectors = FaceDetectors(self)
        self.face_landmark_detectors = FaceLandmarkDetectors(self)
        self.face_masks = FaceMasks(self)
        self.face_restorers = FaceRestorers(self)
        self.face_swappers = FaceSwappers(self)
        self.frame_enhancers = FrameEnhancers(self)
        self.face_editors = FaceEditors(self)

        # Denoiser specific initializations
        self.lp_mask_crop_latent = faceutil.create_faded_inner_mask(size=(64, 64), border_thickness=3, fade_thickness=8, blur_radius=3, device=self.device)
        self.lp_mask_crop_latent = torch.unsqueeze(self.lp_mask_crop_latent, 0) # Shape: [1, 64, 64]
        self.betas_np = np.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=np.float64)**2 # Common linear schedule for 1000 steps
        self.alphas_np = 1.0 - self.betas_np
        self.alphas_cumprod_np = np.cumprod(self.alphas_np, axis=0)
        self.alphas_cumprod_torch = torch.from_numpy(self.alphas_cumprod_np).float().to(self.device)
        self.vae_scale_factor = 0.18215 # Typical LDM VAE scale factor

        self.clip_session = []
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])
        self.mean_lmk = []
        self.anchors  = []
        self.emap = []
        self.LandmarksSubsetIdxs = [
            0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
            40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
            81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
            136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
            161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
            249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
            296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
            336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
            384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
            466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
        ]

        self.normalize = v2.Normalize(mean = [ 0., 0., 0. ],
                                      std = [ 1/1.0, 1/1.0, 1/1.0 ])
        
        self.lp_mask_crop = self.face_editors.lp_mask_crop
        self.lp_lip_array = self.face_editors.lp_lip_array

    # --- Model Loading, Unloading, and Management ---

    def load_model(self, model_name, session_options=None):
        with self.model_lock:
            self.main_window.model_loading_signal.emit()
            # QApplication.processEvents()
            # if not is_file_exists(self.models_path[model_name]):
            #     download_file(model_name, self.models_path[model_name], self.models_data[model_name]['hash'], self.models_data[model_name]['url'])
            if session_options is None:
                model_instance = onnxruntime.InferenceSession(self.models_path[model_name], providers=self.providers)
            else:
                model_instance = onnxruntime.InferenceSession(self.models_path[model_name], sess_options=session_options, providers=self.providers)

            # Check if another thread has already loaded an instance for this model, if yes then delete the current one and return that instead
            if self.models[model_name]:
                del model_instance
                gc.collect()
                return self.models[model_name]
            self.main_window.model_loaded_signal.emit()

            return model_instance

    def load_dfm_model(self, dfm_model):
        with self.model_lock:
            if not self.dfm_models.get(dfm_model):
                self.main_window.model_loading_signal.emit()
                max_models_to_keep = self.main_window.control['MaxDFMModelsSlider']
                total_loaded_models = len(self.dfm_models)
                if total_loaded_models==max_models_to_keep:
                    print("Clearing DFM Model")
                    model_name, model_instance = list(self.dfm_models.items())[0]
                    del model_instance
                    self.dfm_models.pop(model_name)
                    gc.collect()
                try:
                    self.dfm_models[dfm_model] = DFMModel(self.main_window.dfm_models_data[dfm_model], self.providers, self.device)
                except:
                    traceback.print_exc()   
                    self.dfm_models[dfm_model] = None         
                self.main_window.model_loaded_signal.emit()
            return self.dfm_models[dfm_model]


    def load_model_trt(self, model_name, custom_plugin_path=None, precision='fp16', debug=False):
        # self.showModelLoadingProgressBar()
        #time.sleep(0.5)
        self.main_window.model_loading_signal.emit()

        if not os.path.exists(self.models_trt_path[model_name]):
            onnx2trt(onnx_model_path=self.models_path[model_name],
                     trt_model_path=self.models_trt_path[model_name],
                     precision=precision,
                     custom_plugin_path=custom_plugin_path,
                     verbose=False
                    )
        model_instance = TensorRTPredictor(model_path=self.models_trt_path[model_name], custom_plugin_path=custom_plugin_path, pool_size=self.nThreads, device=self.device, debug=debug)

        self.main_window.model_loaded_signal.emit()
        return model_instance

    def delete_models(self):
        for model_name, model_instance in self.models.items():
            del model_instance
            self.models[model_name] = None
        self.clip_session = []
        gc.collect()

    def delete_models_trt(self):
        if TENSORRT_AVAILABLE:
            for model_data in models_trt_list:
                model_name = model_data['model_name']
                if isinstance(self.models_trt[model_name], TensorRTPredictor):
                    # È un'istanza di TensorRTPredictor
                    self.models_trt[model_name].cleanup()
                    del self.models_trt[model_name]
                    self.models_trt[model_name] = None #Model Instance
            gc.collect()

    def delete_models_dfm(self):
        keys_to_remove = []
        for model_name, model_instance in self.dfm_models.items():
            del model_instance
            keys_to_remove.append(model_name)
        
        for model_name in keys_to_remove:
            self.dfm_models.pop(model_name)
        
        self.clip_session = []
        gc.collect()

    def unload_model(self, model_name_to_unload):
        with self.model_lock:
            if model_name_to_unload in self.models and self.models.get(model_name_to_unload) is not None:
                print(f"Unloading model: {model_name_to_unload}")
                del self.models[model_name_to_unload]
                self.models[model_name_to_unload] = None 
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print(f"Model '{model_name_to_unload}' not found or already unloaded.")

    def showModelLoadingProgressBar(self):
        self.main_window.model_load_dialog.show()

    def hideModelLoadProgressBar(self):
        if self.main_window.model_load_dialog:
            self.main_window.model_load_dialog.close()

    def switch_providers_priority(self, provider_name):
        match provider_name:
            case "TensorRT" | "TensorRT-Engine":
                providers = [
                                ('TensorrtExecutionProvider', self.trt_ep_options),
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cuda'
                if version.parse(trt.__version__) < version.parse("10.2.0") and provider_name == "TensorRT-Engine":
                    print("TensorRT-Engine provider cannot be used when TensorRT version is lower than 10.2.0.")
                    provider_name = "TensorRT"

            case "CPU":
                providers = [
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cpu'
            case "CUDA":
                providers = [
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cuda'
            #case _:

        self.providers = providers
        self.provider_name = provider_name
        self.lp_mask_crop = self.lp_mask_crop.to(self.device)

        return self.provider_name

    def set_number_of_threads(self, value):
        self.nThreads = value
        self.delete_models_trt()

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_total = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        memory_used = memory_total[0] - memory_free[0]

        return memory_used, memory_total[0]
    
    def clear_gpu_memory(self):
        self.delete_models()
        self.delete_models_dfm()
        self.delete_models_trt()
        torch.cuda.empty_cache()

    def ensure_denoiser_models_loaded(self):
        """Loads the UNet and VAE models if they are not already loaded."""
        with self.model_lock: # Ensure thread safety
            #print("Ensuring denoiser models (UNet, VAEs) are loaded...")
            unet_model_name = self.main_window.fixed_unet_model_name
            vae_encoder_name = 'RefLDMVAEEncoder'
            vae_decoder_name = 'RefLDMVAEDecoder'

            if not self.models.get(unet_model_name): # Use .get() for safety
                #print(f"  Loading UNet model: {unet_model_name}")
                self.models[unet_model_name] = self.load_model(unet_model_name)
            # else:
                # print(f"  UNet model '{unet_model_name}' already loaded.")

            if not self.models.get(vae_encoder_name):
                #print(f"  Loading VAE Encoder model: {vae_encoder_name}")
                self.models[vae_encoder_name] = self.load_model(vae_encoder_name)
            # else:
                # print(f"  VAE Encoder model '{vae_encoder_name}' already loaded.")

            if not self.models.get(vae_decoder_name):
                #print(f"  Loading VAE Decoder model: {vae_decoder_name}")
                self.models[vae_decoder_name] = self.load_model(vae_decoder_name)
            # else:
                # print(f"  VAE Decoder model '{vae_decoder_name}' already loaded.")
            #print("Denoiser models loading check complete.")

    def unload_denoiser_models(self):
        """Unloads the UNet and VAE models."""
        with self.model_lock: # Ensure thread safety
            print("Unloading denoiser models (UNet, VAEs)...")
            self.unload_model(self.main_window.fixed_unet_model_name)
            self.unload_model('RefLDMVAEEncoder')
            self.unload_model('RefLDMVAEDecoder')
            print("Denoiser models unloaded.")


    def load_inswapper_iss_emap(self, model_name):
        with self.model_lock:
            if not self.models[model_name]:
                self.main_window.model_loading_signal.emit()
                graph = onnx.load(self.models_path[model_name]).graph
                self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
                self.main_window.model_loaded_signal.emit()

    # --- Face Processing Methods ---

    def run_detect(self, img, detect_mode='RetinaFace', max_num=1, score=0.5, input_size=(512, 512), use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        return self.face_detectors.run_detect(img, detect_mode, max_num, score, input_size, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles)
    
    def run_detect_landmark(self, img, bbox, det_kpss, detect_mode='203', score=0.5, from_points=False):
        return self.face_landmark_detectors.run_detect_landmark(img, bbox, det_kpss, detect_mode, score, from_points)

    def get_arcface_model(self, face_swapper_model): 
        if face_swapper_model in arcface_mapping_model_dict:
            return arcface_mapping_model_dict[face_swapper_model]
        else:
            raise ValueError(f"Face swapper model {face_swapper_model} not found.")

    def run_recognize_direct(self, img, kps, similarity_type='Opal', arcface_model='Inswapper128ArcFace'):
        return self.face_swappers.run_recognize_direct(img, kps, similarity_type, arcface_model)

    # --- Swapper Methods ---

    def calc_inswapper_latent(self, source_embedding):
        return self.face_swappers.calc_inswapper_latent(source_embedding)

    def run_inswapper(self, image, embedding, output):
        self.face_swappers.run_inswapper(image, embedding, output)

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        return self.face_swappers.calc_swapper_latent_iss(source_embedding, version)

    def run_iss_swapper(self, image, embedding, output, version="A"):
        self.face_swappers.run_iss_swapper(image, embedding, output, version)

    def calc_swapper_latent_simswap512(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_simswap512(source_embedding)

    def run_swapper_simswap512(self, image, embedding, output):
        self.face_swappers.run_swapper_simswap512(image, embedding, output)

    def calc_swapper_latent_ghost(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_ghost(source_embedding)

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GhostFace-v2'):
        self.face_swappers.run_swapper_ghostface(image, embedding, output, swapper_model)

    def calc_swapper_latent_cscs(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_cscs(source_embedding)

    def run_swapper_cscs(self, image, embedding, output):
        self.face_swappers.run_swapper_cscs(image, embedding, output)

    # --- Frame Enhancer Methods ---

    def run_enhance_frame_tile_process(self, img, enhancer_type, tile_size=256, scale=1):
        return self.frame_enhancers.run_enhance_frame_tile_process(img, enhancer_type, tile_size, scale)

    def run_deoldify_artistic(self, image, output):
        return self.frame_enhancers.run_deoldify_artistic(image, output)

    def run_deoldify_stable(self, image, output):
        return self.frame_enhancers.run_deoldify_artistic(image, output)
    
    def run_deoldify_video(self, image, output):
        return self.frame_enhancers.run_deoldify_video(image, output)
    
    def run_ddcolor_artistic(self, image, output):
        return self.frame_enhancers.run_ddcolor_artistic(image, output)

    def run_ddcolor(self, tensor_gray_rgb, output_ab):
        return self.frame_enhancers.run_ddcolor(tensor_gray_rgb, output_ab)

    # --- Masking Methods ---

    def run_occluder(self, image, output):
        self.face_masks.run_occluder(image, output)

    def run_dfl_xseg(self, image, output):
        self.face_masks.run_dfl_xseg(image, output)

    def run_faceparser(self, image, output):
        self.face_masks.run_faceparser(image, output)

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        return self.face_masks.run_CLIPs(img, CLIPText, CLIPAmount)

    # --- LivePortrait (Face Editor) Methods ---
    
    def lp_motion_extractor(self, img, face_editor_type='Human-Face', **kwargs) -> dict:
        return self.face_editors.lp_motion_extractor(img, face_editor_type, **kwargs)

    def lp_appearance_feature_extractor(self, img, face_editor_type='Human-Face'):
        return self.face_editors.lp_appearance_feature_extractor(img, face_editor_type)

    def lp_retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_retarget_eye(kp_source, eye_close_ratio, face_editor_type)

    def lp_retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_retarget_lip(kp_source, lip_close_ratio, face_editor_type)

    def lp_stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_stitch(kp_source, kp_driving, face_editor_type)

    def lp_stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_stitching(kp_source, kp_driving, face_editor_type)

    def lp_warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_warp_decode(feature_3d, kp_source, kp_driving, face_editor_type)

    # --- Utility and Combined Methods ---

    def findCosineDistance(self, vector1, vector2):
        vector1 = vector1.ravel()
        vector2 = vector2.ravel()
        cos_dist = 1 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) # 2..0
        return 100-cos_dist*50

    def apply_facerestorer(self, swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score):
        return self.face_restorers.apply_facerestorer(swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score)

    def apply_occlusion(self, img, amount):
        return self.face_masks.apply_occlusion(img, amount)
    
    def apply_dfl_xseg(self, img, amount):
        return self.face_masks.apply_dfl_xseg(img, amount)
    
    def apply_face_parser(self, img, parameters):
        return self.face_masks.apply_face_parser(img, parameters)
    
    def apply_face_makeup(self, img, parameters):
        return self.face_editors.apply_face_makeup(img, parameters)
    
    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        return self.face_masks.restore_mouth(img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset)

    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        return self.face_masks.restore_eyes(img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset, eye_spacing_offset)

    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        return self.face_masks.apply_fake_diff(swapped_face, original_face, DiffAmount)

    # --- UNet Denoiser Specific Methods ---

    def apply_denoiser_unet(self, 
                            image_cxhxw_uint8: torch.Tensor, 
                            reference_kv_filename: str, # Path to the .pt file (just filename)
                            use_reference_exclusive_path: bool, # New flag
                            denoiser_mode: str = "Single Step (Fast)", # Still here, but DDIM part removed
                            denoiser_single_step_t: int = 10,
                            base_seed: int = 0
                            ) -> torch.Tensor:
        # Input: CxHxW, uint8, RGB, range [0, 255]
        # Output: CxHxW, uint8, RGB, range [0, 255]

        unet_model_name = self.main_window.fixed_unet_model_name
        vae_encoder_name = 'RefLDMVAEEncoder'
        vae_decoder_name = 'RefLDMVAEDecoder'
        kv_tensor_map_for_this_run: Dict[str, Dict[str, torch.Tensor]] | None = None

        with self.model_lock: # Ensure thread-safe access to model loading and K/V map
            # --- Ensure all denoiser-related models are loaded ---
            self.ensure_denoiser_models_loaded()

            # --- Check if essential models for this function are actually loaded ---
            if not (self.models.get(unet_model_name) and \
                    self.models.get(vae_encoder_name) and \
                    self.models.get(vae_decoder_name)):
                print("Denoiser: Critical models (UNet/VAEs) not loaded after check. Skipping.")
                return image_cxhxw_uint8

            if not reference_kv_filename or reference_kv_filename == "No K/V tensor files found":
                print(f"Denoiser: No K/V tensor file selected ('{reference_kv_filename}'). Skipping.")
                return image_cxhxw_uint8

            # Step 1: Get the master K/V map from MainWindow.
            # This map is now assumed to be loaded/updated by MainWindow.handle_reference_kv_file_change
            # when the UI selection changes. We just read it here.
            master_kv_map_from_main = self.main_window.current_kv_tensors_map
            if master_kv_map_from_main is None or not master_kv_map_from_main:
                print(f"Denoiser: Master K/V map from MainWindow for '{reference_kv_filename}' is None or empty. Skipping.")
                return image_cxhxw_uint8
            
            # Step 2: Check if ModelsProcessor's internal deep copy needs updating
            if self.internal_deep_copied_kv_map is None or \
               self.internal_kv_map_source_filename != reference_kv_filename:

                # Ensure the master_kv_map_from_main actually corresponds to the reference_kv_filename
                # that this worker thread is processing for.
                if self.main_window.previous_kv_file_selection != reference_kv_filename:
                    print(f"Denoiser Warning: Stale K/V map in MainWindow ('{self.main_window.previous_kv_file_selection}') "
                          f"for worker processing '{reference_kv_filename}'. This might happen during rapid K/V file changes. "
                          f"Attempting to use current MainWindow map. If issues persist, re-select K/V file or pause processing.")
                    # If master_kv_map_from_main is for a different file, using it might be wrong.
                    # However, if previous_kv_file_selection *is* reference_kv_filename, then master_kv_map_from_main is correct.

                print(f"ModelsProcessor: Updating internal deep-copied K/V map. Target source file for cache: '{reference_kv_filename}'.")
                try:
                    # Perform the deep copy here, under the lock, to ensure atomicity of cache update.
                    # Tensors are also moved to the correct device.
                    self.internal_deep_copied_kv_map = {
                        layer: {
                            'k': tens_dict['k'].clone().to(self.device), 
                            'v': tens_dict['v'].clone().to(self.device)
                        }
                        for layer, tens_dict in master_kv_map_from_main.items()
                        if tens_dict and isinstance(tens_dict.get('k'), torch.Tensor) and isinstance(tens_dict.get('v'), torch.Tensor)
                    }
                    self.internal_kv_map_source_filename = reference_kv_filename
                    # print(f"ModelsProcessor: Internal K/V map cache updated. {len(self.internal_deep_copied_kv_map)} layers.")
                except Exception as e:
                    print(f"Denoiser: Error deep copying K/V map into internal cache for '{reference_kv_filename}': {e}. Skipping.")
                    self.internal_deep_copied_kv_map = None # Invalidate cache on error
                    self.internal_kv_map_source_filename = None
                    return image_cxhxw_uint8
            kv_tensor_map_for_this_run = self.internal_deep_copied_kv_map

        # --- Actual denoise operation using the obtained kv_tensor_map_for_this_run ---
        # The ONNX model runs (face_restorers.run_vae_encoder, etc.) use self.models,
        # which were ensured to be loaded under the lock. InferenceSession.run is generally thread-safe.

        target_proc_dim = 512
        _, h_input, w_input = image_cxhxw_uint8.shape
        old_deterministic_state = torch.are_deterministic_algorithms_enabled()
        
        if h_input != target_proc_dim or w_input != target_proc_dim:
            resize_transform = v2.Resize((target_proc_dim, target_proc_dim), 
                                          interpolation=v2.InterpolationMode.BILINEAR, 
                                          antialias=True)
            image_to_process_cxhxw_uint8 = resize_transform(image_cxhxw_uint8)
        else:
            image_to_process_cxhxw_uint8 = image_cxhxw_uint8

        torch.manual_seed(base_seed) # For noise generation consistency
        
        h_proc, w_proc = image_to_process_cxhxw_uint8.shape[1], image_to_process_cxhxw_uint8.shape[2]
        image_normalized_bchw = (image_to_process_cxhxw_uint8.float() / 127.5) - 1.0
        image_normalized_bchw = image_normalized_bchw.unsqueeze(0).contiguous()

        latent_h, latent_w = h_proc // 8, w_proc // 8
        encoded_latent_8_channel = torch.empty((1, 8, latent_h, latent_w), 
                                               dtype=torch.float32, 
                                               device=self.device).contiguous()

        self.face_restorers.run_vae_encoder(image_normalized_bchw, encoded_latent_8_channel)

        lq_latent_scaled_for_unet = encoded_latent_8_channel * self.vae_scale_factor

        is_ref_flag_tensor = torch.tensor([False], dtype=torch.bool, device=self.device).contiguous() # Shape (1,)
        use_reference_exclusive_path_tensor = torch.tensor([use_reference_exclusive_path], dtype=torch.bool, device=self.device).contiguous() # Shape (1,)
         # pred_x0_unscaled was previously initialized here but not directly used before re-assignment
        # It's better to calculate it directly when needed.

        # --- Validate kv_tensor_map_for_this_run (points to self.internal_deep_copied_kv_map) ---
        if kv_tensor_map_for_this_run: # If not None and not empty
            is_kv_map_valid = True
            for layer_name, kv_dict in kv_tensor_map_for_this_run.items():
                # Check if 'k' or 'v' are missing, or are not torch.Tensor objects
                k_tensor = kv_dict.get('k')
                v_tensor = kv_dict.get('v')
                if not isinstance(kv_dict, dict) or \
                   k_tensor is None or not isinstance(k_tensor, torch.Tensor) or \
                   v_tensor is None or not isinstance(v_tensor, torch.Tensor):
                    print(f"Denoiser: Invalid K/V entry for layer '{layer_name}' in map for '{reference_kv_filename}'. K type: {type(k_tensor)}, V type: {type(v_tensor)}. Skipping UNet.")
                    is_kv_map_valid = False
                    break
            if not is_kv_map_valid:
                return image_cxhxw_uint8 # Return original if map content is bad
        else: # kv_tensor_map_for_this_run is None or empty
            print(f"Denoiser: K/V map is None or empty for '{reference_kv_filename}' before UNet call. Skipping.")
            return image_cxhxw_uint8

        x0_unscaled_for_noise_addition = encoded_latent_8_channel 
        timesteps_tensor_unet = torch.tensor([denoiser_single_step_t], dtype=torch.int64, device=self.device)

        alpha_t_val = self.alphas_cumprod_np[denoiser_single_step_t]
        sqrt_alpha_bar_t = torch.sqrt(torch.tensor(alpha_t_val, device=self.device, dtype=torch.float32))
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - torch.tensor(alpha_t_val, device=self.device, dtype=torch.float32))

        torch.manual_seed(base_seed + denoiser_single_step_t) # Seed for this specific noise sample
        noise_sample = torch.randn_like(x0_unscaled_for_noise_addition)
        xt_noisy_unscaled_8_channel = x0_unscaled_for_noise_addition * sqrt_alpha_bar_t + noise_sample * sqrt_one_minus_alpha_bar_t

        unet_input_16_channel = torch.cat((xt_noisy_unscaled_8_channel, lq_latent_scaled_for_unet), dim=1)
        predicted_noise_from_unet = torch.empty((1, 8, latent_h, latent_w), dtype=torch.float32, device=self.device).contiguous()
        
        self.face_restorers.run_ref_ldm_unet(
            x_noisy_plus_lq_latent=unet_input_16_channel,
            timesteps_tensor=timesteps_tensor_unet,
            is_ref_flag_tensor=is_ref_flag_tensor,
            use_reference_exclusive_path_globally_tensor=use_reference_exclusive_path_tensor,
            kv_tensor_map=kv_tensor_map_for_this_run, 
            output_unet_tensor=predicted_noise_from_unet
        )
        
        pred_x0_unscaled_final = (xt_noisy_unscaled_8_channel - sqrt_one_minus_alpha_bar_t * predicted_noise_from_unet) / sqrt_alpha_bar_t
        
        torch.use_deterministic_algorithms(old_deterministic_state)

        latent_for_vae_decoder = pred_x0_unscaled_final # Use the correctly calculated x0
        decoded_image_normalized_bchw = torch.empty((1, 3, h_proc, w_proc), 
                                                    dtype=torch.float32, 
                                                    device=self.device).contiguous()

        self.face_restorers.run_vae_decoder(latent_for_vae_decoder, decoded_image_normalized_bchw)

        denoised_image_cxhxw_uint8 = ((decoded_image_normalized_bchw.squeeze(0) + 1.0) * 127.5).clamp(0, 255).byte()
        
        # Return the raw denoised image. Masking will be handled by the caller (e.g., swap_core).
        return denoised_image_cxhxw_uint8