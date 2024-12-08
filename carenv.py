import random
import time
import numpy as np
import math
import cv2
import gymnasium as gym  # Updated from gym to gymnasium
from gymnasium import spaces
import carla

SECONDS_PER_EPISODE = 25

N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320

FIXED_DELTA_SECONDS = 0.2

SHOW_PREVIEW = True


class CarEnv(gym.Env):  # Updated from gym to gymnasium
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = WIDTH
    im_height = HEIGHT
    front_camera = None
    CAMERA_POS_Z = 1.3
    CAMERA_POS_X = 1.4

    def __init__(self):
        super(CarEnv, self).__init__()
        # Define action space: 9 steering values x 4 throttle/brake values = 36 discrete actions
        self.action_space = spaces.Discrete(36)
        
        # Observation space for images normalized to 0..1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32
        )
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()

        # Configure CARLA world settings
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = True
        self.settings.synchronous_mode = False
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def cleanup(self):
        for sensor in self.world.get_actors().filter("*sensor*"):
            sensor.destroy()
        for actor in self.world.get_actors().filter("*vehicle*"):
            actor.destroy()
        cv2.destroyAllWindows()

    def step(self, action):
        self.step_counter += 1
        
        # Map the discrete action to steer and throttle/brake
        steer_idx = action // 4  # Integer division for steering index
        throttle_idx = action % 4  # Remainder for throttle/brake index

        # Map steering actions
        steer_map = [-0.9, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.9]
        steer = steer_map[steer_idx]

        # Map throttle/brake actions
        throttle_map = [
            (0.0, 1.0),  # Brake
            (0.3, 0.0),  # Slow throttle
            (0.7, 0.0),  # Medium throttle
            (1.0, 0.0),  # Full throttle
        ]
        throttle_val, brake_val = throttle_map[throttle_idx]

        # Apply control to the vehicle
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle_val, steer=steer, brake=brake_val)
        )

        # Optional - print steer and throttle every 50 steps
        if self.step_counter % 50 == 0:
            print("steer input:", steer, ", throttle:", throttle_val)

        # Calculate velocity and distance traveled
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        distance_travelled = self.initial_location.distance(self.vehicle.get_location())

        # Display the camera feed
        if self.SHOW_CAM:
            cv2.imshow("Sem Camera", self.front_camera)
            cv2.waitKey(1)

        # Steering lock detection
        lock_duration = 0
        if not self.steering_lock:
            if steer < -0.6 or steer > 0.6:
                self.steering_lock = True
                self.steering_lock_start = time.time()
        else:
            if steer < -0.6 or steer > 0.6:
                lock_duration = time.time() - self.steering_lock_start

        # Calculate reward
        reward = 0
        done = False

        if len(self.collision_hist) != 0:  # Collision penalty
            done = True
            reward -= 300
            self.cleanup()

        if lock_duration > 3:  # Steering lock penalty
            reward -= 150
            done = True
            self.cleanup()
        elif lock_duration > 1:
            reward -= 20

        if kmh < 10:  # Speed penalties and rewards
            reward -= 3
        elif kmh < 15:
            reward -= 1
        elif kmh > 40:
            reward -= 10
        else:
            reward += 1

        if distance_travelled < 30:  # Distance rewards
            reward -= 1
        elif distance_travelled < 50:
            reward += 1
        else:
            reward += 2

        if self.episode_start + SECONDS_PER_EPISODE < time.time():  # Episode end
            done = True
            self.cleanup()

        return self.front_camera / 255.0, reward, done, False, {}

    def reset(self, seed=None):
        self.collision_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = None
        while self.vehicle is None:
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            except:
                pass
        self.actor_list.append(self.vehicle)
        self.initial_location = self.vehicle.get_location()

        self.sem_cam = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", "90")

        camera_init_trans = carla.Transform(
            carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X)
        )
        self.sensor = self.world.spawn_actor(
            self.sem_cam, camera_init_trans, attach_to=self.vehicle
        )
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        if self.SHOW_CAM:  # Show camera feed
            cv2.namedWindow("Sem Camera", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Sem Camera", self.front_camera)
            cv2.waitKey(1)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(
            colsensor, camera_init_trans, attach_to=self.vehicle
        )
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None
        self.step_counter = 0
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera / 255.0, {}

    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]  # Ignore alpha
        self.front_camera = i

    def collision_data(self, event):
        self.collision_hist.append(event)