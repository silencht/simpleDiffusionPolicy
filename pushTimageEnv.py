from gym import spaces
from pushTenv import PushTEnv
import numpy as np
import cv2

# Why must the implementation of pushTimageEnv.py and pushTenv.py be separated???

class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None
    
    def _get_obs(self):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()
        
        return self.render_cache

# Env Demo
if __name__ == "__main__":
    # Standard Gym Env (0.21.0 API)
    # 0. create env object
    env = PushTImageEnv()
    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(1000)
    # 2. must reset before use
    obs, info = env.reset()
    # 3. 2D positional action space [0,512]
    action = env.action_space.sample()
    # 4. Standard gym step method
    obs, reward, terminated, truncated, info = env.step(action)

    # prints and explains each dimension of the observation and action vectors
    with np.printoptions(precision=4, suppress=True, threshold=5):
        print("action: ", action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("terminated: ", terminated)
        print("truncated: ", truncated)
        print("info: ", info)
        
    # visualization
    image_obs = obs['image']
    image_to_show = (image_obs * 255).astype(np.uint8).transpose(1, 2, 0)
    cv2.namedWindow('Environment Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Environment Image', image_to_show)
    cv2.waitKey(0)

    obs, reward, terminated, truncated, info = env.step(action)
    with np.printoptions(precision=4, suppress=True, threshold=5):
        print("action: ", action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("terminated: ", terminated)
        print("truncated: ", truncated)
        print("info: ", info)
        # visualization
    image_obs = obs['image']
    image_to_show = (image_obs * 255).astype(np.uint8).transpose(1, 2, 0)
    cv2.namedWindow('Environment Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Environment Image', image_to_show)
    cv2.waitKey(0)