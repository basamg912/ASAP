import math
import queue

import torch

from humanoidverse.utils.math import quat_apply_yaw


_NO_REQUEST = object()


class KeyboardBasePushController:
    """Queue keyboard requests and apply fixed-duration forces in physics steps."""

    DIRECTIONS = {
        "forward": (1.0, 0.0, 0.0),
        "backward": (-1.0, 0.0, 0.0),
        "left": (1.0, 1.0, 0.0),
        "right": (1.0, -1.0, 0.0),
    }

    def __init__(self, env, force_newtons, duration_seconds):
        if force_newtons <= 0.0:
            raise ValueError("keyboard push force must be positive")
        if duration_seconds <= 0.0:
            raise ValueError("keyboard push duration must be positive")

        self.env = env
        self.force_newtons = float(force_newtons)
        self.duration_seconds = float(duration_seconds)
        self.duration_steps = max(
            1, math.ceil(self.duration_seconds / self.env.simulator.sim_dt)
        )
        self._requests = queue.SimpleQueue()
        self._active_direction = None
        self._remaining_steps = 0
        self._owns_force = False
        self._last_episode_length = int(self.env.episode_length_buf[0].item())
        self._installed = False

    def request(self, direction):
        if direction not in self.DIRECTIONS:
            raise ValueError(f"unknown keyboard push direction: {direction}")
        self._requests.put(direction)

    def cancel(self):
        self._requests.put(None)

    def install(self):
        """Run the controller after torques are set and before each physics step."""
        if self._installed:
            return

        original_apply_force = self.env._apply_force_in_physics_step

        def apply_force_with_keyboard_push():
            original_apply_force()
            self.apply_at_physics_step()

        self.env._apply_force_in_physics_step = apply_force_with_keyboard_push
        self._installed = True

    def apply_at_physics_step(self):
        episode_length = int(self.env.episode_length_buf[0].item())
        if episode_length < self._last_episode_length:
            self._stop()
        self._last_episode_length = episode_length

        latest_request = self._pop_latest_request()
        if latest_request is None:
            if self._remaining_steps > 0 or self._owns_force:
                self._stop()
        elif latest_request is not _NO_REQUEST:
            self._active_direction = latest_request
            self._remaining_steps = self.duration_steps

        if self._remaining_steps > 0:
            direction = self.DIRECTIONS[self._active_direction]
            local_force = torch.tensor(
                direction,
                device=self.env.base_quat.device,
                dtype=self.env.base_quat.dtype,
            ).unsqueeze(0)
            local_force *= self.force_newtons
            world_force = quat_apply_yaw(self.env.base_quat[:1], local_force)[0]
            self.env.simulator.set_root_external_force(world_force)
            self._remaining_steps -= 1
            self._owns_force = True
        elif self._owns_force:
            self._stop()

    def _pop_latest_request(self):
        latest_request = _NO_REQUEST
        while True:
            try:
                latest_request = self._requests.get_nowait()
            except queue.Empty:
                return latest_request

    def _stop(self):
        if self._owns_force:
            self.env.simulator.set_root_external_force((0.0, 0.0, 0.0))
        self._active_direction = None
        self._remaining_steps = 0
        self._owns_force = False
