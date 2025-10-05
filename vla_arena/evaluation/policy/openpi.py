import websockets
import functools
import time
import collections
from typing import Dict, Tuple
import websockets.sync.client
from typing_extensions import override
import msgpack
import numpy as np
from vla_arena.evaluation.policy.base import Policy, PolicyRegistry
from vla_arena.evaluation.utils import normalize_gripper_action, invert_gripper_action
from vla_arena.evaluation.utils import read_eval_cfgs, resize_image

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
    if den > 0.0:
        quat[0] /= den
        quat[1] /= den
        quat[2] /= den
    return quat[:3]


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)

@PolicyRegistry.register("openpi")
class OpenPI(Policy):
    def __init__(self, 
                 host="0.0.0.0",
                 port=8000,
                 replan_steps=4,
                 eval_cfgs_path='../../configs/evaluation/openpi.yaml',
                 **kwargs):
        super().__init__(**kwargs)
        self.eval_cfgs = read_eval_cfgs(self.name, eval_cfgs_path)
        self.replan_steps = replan_steps
        self.action_plan = collections.deque(maxlen=replan_steps)
        self.timestep = 0
        self._uri = f"ws://{host}:{port}"
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()
    
    def _process_observation(self, obs, **kwargs):
        return {
            "observation/image": obs["agentview_image"][::-1, ::-1, :],
            "observation/wrist_image": obs["robot0_eye_in_hand_image"][::-1, ::-1, :],
            "observation/state": np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"], np.array([obs["robot0_gripper_open"]])]),
            "prompt": self.instruction,
        }

    def _wait_for_server(self):
        while True:
            try:
                ws = websockets.sync.client.ClientConnection(self._uri)
                return ws, ws.metadata
            except Exception as e:
                print(f"Failed to connect to OpenPI server: {e}")
                time.sleep(1)

    def predict(self, observation, **kwargs):
        if self.timestep%self.replan_steps==0:    
            input = self._process_observation(observation, **kwargs)
            action_chunk = self.infer(input)["actions"]
            self.action_plan.extend(action_chunk[: self.replan_steps])
        raw_action = self.action_plan.popleft()
        return self._process_action(raw_action)
    
    def infer(self, obs):
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)