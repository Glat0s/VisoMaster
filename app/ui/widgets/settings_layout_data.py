from app.ui.widgets.actions import control_actions
import cv2
from app.helpers.typing_helper import LayoutDictTypes
SETTINGS_LAYOUT_DATA: LayoutDictTypes = {
    'Appearance': {
        'ThemeSelection': {
            'level': 1,
            'label': 'Theme',
            'options': ['Dark', 'Dark-Blue', 'Light'],
            'default': 'Dark',
            'help': 'Select the theme to be used',
            'exec_function': control_actions.change_theme,
            'exec_function_args': [],
        },
    },
    'General': {
        'ProvidersPrioritySelection': {
            'level': 1,
            'label': 'Providers Priority',
            'options': ['CUDA', 'TensorRT', 'TensorRT-Engine', 'CPU'],
            'default': 'CUDA',
            'help': 'Select the providers priority to be used with the system.',
            'exec_function': control_actions.change_execution_provider,
            'exec_function_args': [],
        },
        'nThreadsSlider': {
            'level': 1,
            'label': 'Number of Threads',
            'min_value': '1',
            'max_value': '30',
            'default': '2',
            'step': 1,
            'help': 'Set number of execution threads while playing and recording. Depends strongly on GPU VRAM.',
            'exec_function': control_actions.change_threads_number,
            'exec_function_args': [],
        },
    },
    'Video Playback Settings': {
        'VideoPlaybackCustomFpsToggle': {
            'level': 1,
            'label': 'Set Custom Video Playback FPS',
            'default': False,
            'help': 'Manually set the FPS to be used when playing the video',
            'exec_function': control_actions.set_video_playback_fps,
            'exec_function_args': [],
        },
        'VideoPlaybackCustomFpsSlider': {
            'level': 2,
            'label': 'Video Playback FPS',
            'min_value': '1',
            'max_value': '120',
            'default': '30',
            'parentToggle': 'VideoPlaybackCustomFpsToggle',
            'requiredToggleValue': True,
            'step': 1,
            'help': 'Set the maximum FPS of the video when playing'
        },
        'LiveSoundSpeedDecimalSlider': {
            'level': 1,
            'label': 'Audio Playback Speed',
            'min_value': '0.50',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'help': '[Experimental] Set the playback speed of the audio, when Live Sound is enabled'
        },
    },
    'Auto Swap':{
        'AutoSwapToggle': {
            'level': 1,
            'label': 'Auto Swap',
            'default': False,
            'help': 'Automatically Swap all faces using selected Source Faces/Embeddings when loading an video/image file'
        },
    },
    'Detectors': {
        'DetectorModelSelection': {
            'level': 1,
            'label': 'Face Detect Model',
            'options': ['RetinaFace', 'Yolov8', 'SCRFD', 'Yunet'],
            'default': 'RetinaFace',
            'help': 'Select the face detection model to use for detecting faces in the input image or video.'
        },
        'DetectorScoreSlider': {
            'level': 1,
            'label': 'Detect Score',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'help': 'Set the confidence score threshold for face detection. Higher values ensure more confident detections but may miss some faces.'
        },
        'MaxFacesToDetectSlider': {
            'level': 1,
            'label': 'Max No of Faces to Detect',
            'min_value': '1',
            'max_value': '50',
            'default': '20',
            'step': 1,     
            'help': 'Set the maximum number of faces to detect in a frame'
   
        },
        'AutoRotationToggle': {
            'level': 1,
            'label': 'Auto Rotation',
            'default': False,
            'help': 'Automatically rotate the input to detect faces in various orientations.'
        },
        'ManualRotationEnableToggle': {
            'level': 1,
            'label': 'Manual Rotation',
            'default': False,
            'help': 'Rotate the face detector to better detect faces at different angles.'
        },
        'ManualRotationAngleSlider': {
            'level': 2,
            'label': 'Rotation Angle',
            'min_value': '0',
            'max_value': '270',
            'default': '0',
            'step': 90,
            'parentToggle': 'ManualRotationEnableToggle',
            'requiredToggleValue': True,
            'help': 'Set this to the angle of the input face angle to help with laying down/upside down/etc. Angles are read clockwise.'
        },
        'LandmarkDetectToggle': {
            'level': 1,
            'label': 'Enable Landmark Detection',
            'default': False,
            'help': 'Enable or disable facial landmark detection, which is used to refine face alignment.'
        },
        'LandmarkDetectModelSelection': {
            'level': 2,
            'label': 'Landmark Detect Model',
            'options': ['5', '68', '3d68', '98', '106', '203', '478'],
            'default': '203',
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Select the landmark detection model, where different models detect varying numbers of facial landmarks.'
        },
        'LandmarkDetectScoreSlider': {
            'level': 2,
            'label': 'Landmark Detect Score',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Set the confidence score threshold for facial landmark detection.'
        },
        'DetectFromPointsToggle': {
            'level': 2,
            'label': 'Detect From Points',
            'default': False,
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Enable detection of faces from specified landmark points.'
        },
        'ShowLandmarksEnableToggle': {
            'level': 1,
            'label': 'Show Landmarks',
            'default': False,
            'help': 'Show Landmarks in realtime.'
        },
        'ShowAllDetectedFacesBBoxToggle': {
            'level': 1,
            'label': 'Show Bounding Boxes',
            'default': False,
            'help': 'Draw bounding boxes to all detected faces in the frame'
        }
    },
    'DFM Settings':{
        'MaxDFMModelsSlider':{
            'level': 1,
            'label': 'Maximum DFM Models to use',
            'min_value': '1',
            'max_value': '5',
            'default': '1',
            'step': 1,
            'help': "Set the maximum number of DFM Models to keep in memory at a time. Set this based on your GPU's VRAM",
        }
    },
    'Frame Enhancer':{
        'FrameEnhancerEnableToggle':{
            'level': 1,
            'label': 'Enable Frame Enhancer',
            'default': False,
            'help': 'Enable frame enhancement for video inputs to improve visual quality.'
        },
        'FrameEnhancerTypeSelection':{
            'level': 2,
            'label': 'Frame Enhancer Type',
            'options': ['RealEsrgan-x2-Plus', 'RealEsrgan-x4-Plus', 'RealEsr-General-x4v3', 'BSRGan-x2', 'BSRGan-x4', 'UltraSharp-x4', 'UltraMix-x4', 'DDColor-Artistic', 'DDColor', 'DeOldify-Artistic', 'DeOldify-Stable', 'DeOldify-Video'],
            'default': 'RealEsrgan-x2-Plus',
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the type of frame enhancement to apply, based on the content and resolution requirements.'
        },
        'FrameEnhancerBlendSlider': {
            'level': 2,
            'label': 'Blend',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blends the enhanced results back into the original frame.'
        },
    },
    'UNet Denoiser': {
        'DenoiserUNetEnableToggle': {
            'level': 1,
            'widget_type': 'ToggleButton',
            'label': 'Enable UNet Denoiser',
            'control_name': 'DenoiserUNetEnableToggle',
            'default': False,
            'help': 'Enable UNet-based image denoising. This is applied to the 512x512 aligned/swapped face before other restorers.'
        },
        'DenoiserUNetModelSelection': {
            'level': 1,
            'widget_type': 'SelectionBox',
            'label': 'UNet Denoiser Model',
            'control_name': 'DenoiserUNetModelSelection',
            'options': [],
            'default': "",
            'condition_control': 'DenoiserUNetEnableToggle',
            'condition_value': True,
            'help': 'Select the UNet denoiser ONNX model file. Models should be placed in the model_assets folder and start with "ref_ldm_unet_".'
        },
        'DenoiserModeSelection': {
            'level': 2,
            'widget_type': 'SelectionBox',
            'label': 'Denoiser Mode',
            'control_name': 'DenoiserModeSelection',
            'options': ["Single Step (Fast)", "DDIM Loop (Quality)"],
            'default': "Single Step (Fast)",
            'parentToggle': 'DenoiserUNetEnableToggle',
            'requiredToggleValue': True,
            'help': 'Choose denoising mode. Single Step is faster, DDIM Loop may offer better quality at the cost of speed.'
        },
        'DenoiserSingleStepTimestepSlider': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'Single Step Timestep (t)',
            'control_name': 'DenoiserSingleStepTimestepSlider',
            'min_value': '1', 'max_value': '200', 'default': '10', 'step': 1,
            'parentSelection': 'DenoiserModeSelection',
            'requiredSelectionValue': "Single Step (Fast)",
            'help': 'Timestep for single-step denoising. Lower values mean less noise added/removed (subtler effect).'
        },
        'DenoiserDDIMStepsSlider': {
            'level': 3,
            'widget_type': 'ParameterSlider',
            'label': 'DDIM Steps',
            'control_name': 'DenoiserDDIMStepsSlider',
            'min_value': '10', 'max_value': '200', 'default': '50', 'step': 1,
            'parentSelection': 'DenoiserModeSelection',
            'requiredSelectionValue': "DDIM Loop (Quality)",
            'help': 'Number of inference steps for DDIM sampler.'
        },
        'DenoiserDDIMEtaDecimalSlider': {
            'level': 3,
            'widget_type': 'ParameterDecimalSlider',
            'label': 'DDIM Eta',
            'control_name': 'DenoiserDDIMEtaDecimalSlider',
            'min_value': '0.0', 'max_value': '1.0', 'default': '0.0', 'step': 0.1, 'decimals': 1,
            'parentSelection': 'DenoiserModeSelection',
            'requiredSelectionValue': "DDIM Loop (Quality)",
            'help': 'DDIM eta (0.0 for deterministic DDIM, >0 for DDPM-like stochasticity).'
        },
        'DenoiserBaseSeedSlider': {
            'level': 2, # Same level as DenoiserModeSelection
            'widget_type': 'ParameterSlider',
            'label': 'Base Seed',
            'control_name': 'DenoiserBaseSeedSlider',
            'min_value': '0', 'max_value': '999999', 'default': '0', 'step': 1,
            'parentToggle': 'DenoiserUNetEnableToggle', # Depends on the main denoiser toggle
            'requiredToggleValue': True,
            'help': 'Set a fixed base seed for the denoiser. This seed will be used for all frames and both denoiser passes (if applicable) to ensure consistent noise patterns.'
        },
        'DenoiserAfterRestorersToggle': {
            'level': 2, # Same level as DenoiserModeSelection
            'widget_type': 'ToggleButton',
            'label': 'Apply After Restorers',
            'control_name': 'DenoiserAfterRestorersToggle',
            'default': False,
            'parentToggle': 'DenoiserUNetEnableToggle', # Depends on the main denoiser toggle
            'requiredToggleValue': True,
            'help': 'Apply the UNet Denoiser again after face restorers have been applied. Uses the same UNet model and step settings.'
        }
    },
    'Webcam Settings': {
        'WebcamMaxNoSelection': {
            'level': 2,
            'label': 'Webcam Max No',
            'options': ['1', '2', '3', '4', '5', '6'],
            'default': '1',
            'help': 'Select the maximum number of webcam streams to allow for face swapping.'
        },
        'WebcamBackendSelection': {
            'level': 2,
            'label': 'Webcam Backend',
            'options': ['Default', 'DirectShow', 'MSMF', 'V4L', 'V4L2', 'GSTREAMER'],
            'default': 'Default',
            'help': 'Choose the backend for accessing webcam input.'
        },
        'WebcamMaxResSelection': {
            'level': 2,
            'label': 'Webcam Resolution',
            'options': ['480x360', '640x480', '1280x720', '1920x1080', '2560x1440', '3840x2160'],
            'default': '1280x720',
            'help': 'Select the maximum resolution for webcam input.'
        },
        'WebCamMaxFPSSelection': {
            'level': 2,
            'label': 'Webcam FPS',
            'options': ['23', '30', '60'],
            'default': '30',
            'help': 'Set the maximum frames per second (FPS) for webcam input.'
        },
    },
    'Virtual Camera': {
        'SendVirtCamFramesEnableToggle': {
            'level': 1,
            'label': 'Send Frames to Virtual Camera',
            'default': False,
            'help': 'Send the swapped video/webcam output to virtual camera for using in external applications',
            'exec_function': control_actions.toggle_virtualcam,
            'exec_function_args': [],
        },
        'VirtCamBackendSelection': {
            'level': 1,
            'label': 'Virtual Camera Backend',
            'options': ['obs', 'unitycapture'],
            'default': 'obs',
            'help': 'Choose the backend based on the Virtual Camera you have set up',
            'parentToggle': 'SendVirtCamFramesEnableToggle',
            'requiredToggleValue': True,
            'exec_function': control_actions.enable_virtualcam,
            'exec_funtion_args': [],
        },
    },
    'Face Recognition': {
        'RecognitionModelSelection': {
            'level': 1,
            'label': 'Recognition Model',
            'options': ['Inswapper128ArcFace', 'SimSwapArcFace', 'GhostArcFace', 'CSCSArcFace'],
            'default': 'Inswapper128ArcFace',
            'help': 'Choose the ArcFace model to be used for comparing the similarity of faces.'
        },
        'SimilarityTypeSelection': {
            'level': 1,
            'label': 'Swapping Similarity Type',
            'options': ['Opal', 'Pearl', 'Optimal'],
            'default': 'Opal',
            'help': 'Choose the type of similarity calculation for face detection and matching during the face swapping process.'
        },
    },
    'Embedding Merge Method':{
        'EmbMergeMethodSelection':{
            'level': 1,
            'label': 'Embedding Merge Method',
            'options': ['Mean','Median'],
            'default': 'Mean',
            'help': 'Select the method to merge facial embeddings. "Mean" averages the embeddings, while "Median" selects the middle value, providing more robustness to outliers.'
        }
    },
    'Media Selection':{
        'TargetMediaFolderRecursiveToggle':{
            'level': 1,
            'label': 'Target Media Include Subfolders',
            'default': False,
            'help': 'Include all files from Subfolders when choosing Target Media Folder'
        },
        'InputFacesFolderRecursiveToggle':{
            'level': 1,
            'label': 'Input Faces Include Subfolders',
            'default': False,
            'help': 'Include all files from Subfolders when choosing Input Faces Folder'
        }
    }
}

CAMERA_BACKENDS = {
    'Default': cv2.CAP_ANY,
    'DirectShow': cv2.CAP_DSHOW,
    'MSMF': cv2.CAP_MSMF,
    'V4L': cv2.CAP_V4L,
    'V4L2': cv2.CAP_V4L2,
    'GSTREAMER': cv2.CAP_GSTREAMER,
}