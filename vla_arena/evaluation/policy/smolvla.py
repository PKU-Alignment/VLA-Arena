import numpy as np
import torch
import math
from vla_arena.evaluation.policy.base import Policy, PolicyRegistry
from vla_arena.evaluation.policy.modeling_smolvla.modeling_smolvla import SmolVLAPolicy
from vla_arena.evaluation.utils import normalize_gripper_action, invert_gripper_action
from vla_arena.evaluation.utils import read_eval_cfgs

def quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
    Copied from robosuite transform_utils.
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

@PolicyRegistry.register("smolvla")
class SmolVLA(Policy):
    """
    SmolVLA policy implementation for VLA Arena evaluation.
    """
    
    def __init__(self, 
                 model_ckpt,
                 device="cuda",
                 eval_cfgs_path='../../configs/evaluation/smolvla.yaml',
                 **kwargs):
        """
        Initialize SmolVLA policy.
        
        Args:
            model_ckpt: Path to the pretrained SmolVLA model (HuggingFace Hub or local)
            device: Device to run the model on
            **kwargs: Additional arguments
        """
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        
        # Load the pretrained SmolVLA model
        print(f"Loading SmolVLA model from: {model_ckpt}")
        self.eval_cfgs = read_eval_cfgs(self.name, eval_cfgs_path)
        model = SmolVLAPolicy.from_pretrained(model_ckpt)
        model.to(device)
        model.eval()
        
        self.device = device
        self.instruction = None

        # Store instruction if provided in kwargs
        if 'instruction' in kwargs:
            self.instruction = kwargs['instruction']
        
        # Call parent constructor
        super().__init__(model)
        
        print(f"SmolVLA model loaded successfully on {device}")
    
    def reset(self, instruction):
        """
        Reset the policy with a new instruction.
        
        Args:
            instruction: Task instruction/description
        """
        self.instruction = instruction
        # Reset the internal state of the model
        if hasattr(self.model, 'reset'):
            self.model.reset()
    
    def predict(self, obs, **kwargs):
        """
        Predict action given observation.
        
        Args:
            obs: Observation dictionary containing:
                - agentview_image: RGB image from agent view camera
                - robot0_eye_in_hand_image: RGB image from wrist camera (if available)
                - robot0_eef_pos: End-effector position
                - robot0_eef_quat: End-effector quaternion
                - robot0_gripper_qpos: Gripper position
            **kwargs: Additional arguments
            
        Returns:
            action: 7-dimensional action array [x, y, z, rx, ry, rz, gripper]
        """
        # Process observation
        processed_obs = self._prepare_observation(obs)
        
        # Get action from model
        with torch.inference_mode():
            action_tensor = self.model.select_action(processed_obs)
        
        # Convert to numpy and process
        action = action_tensor.cpu().numpy()[0]
        
        # Process the action (normalize gripper and invert if needed)
        action = self._process_action(action)
        
        return action
    
    def _prepare_observation(self, obs):
        """
        Prepare observation dictionary for SmolVLA model.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            processed_obs: Observation dictionary formatted for SmolVLA
        """
        # Rotate images 180 degrees to match training preprocessing
        agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        
        # Check if wrist camera image is available
        if "robot0_eye_in_hand_image" in obs:
            wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        else:
            # If no wrist camera, use agentview as fallback or create dummy
            wrist_image = agentview_image.copy()
        
        # Prepare state vector
        state = np.concatenate([
            obs["robot0_eef_pos"],  # 3D position
            quat2axisangle(obs["robot0_eef_quat"]),  # 3D rotation (axis-angle)
            obs["robot0_gripper_qpos"],  # Gripper state
        ])
        
        # Create observation dictionary for SmolVLA
        processed_obs = {
            "observation.images.image": torch.from_numpy(agentview_image / 255.0)
                .permute(2, 0, 1)  # HWC -> CHW
                .to(torch.float32)
                .to(self.device)
                .unsqueeze(0),  # Add batch dimension
            
            "observation.images.wrist_image": torch.from_numpy(wrist_image / 255.0)
                .permute(2, 0, 1)
                .to(torch.float32)
                .to(self.device)
                .unsqueeze(0),
            
            "observation.state": torch.from_numpy(state)
                .to(torch.float32)
                .to(self.device)
                .unsqueeze(0),
            
            "task": self.instruction if self.instruction else "manipulate objects"
        }
        
        return processed_obs
    
    def _process_action(self, action):
        """
        Process the raw action output from the model.
        
        Args:
            action: Raw action from model
            
        Returns:
            action: Processed action ready for environment
        """
        # Normalize gripper action from [0,1] to [-1,1]
        action = normalize_gripper_action(action, binarize=False)
        
        # Invert gripper action if needed (environment specific)
        action = invert_gripper_action(action)
        
        return action
    
    @property
    def name(self):
        """Return the policy name."""
        return "SmolVLA"
    
    @property
    def control_mode(self):
        """
        Return the control mode of the policy.
        SmolVLA predicts end-effector actions.
        """
        return "ee"