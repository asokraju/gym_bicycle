# Core Library
import logging
import gym
# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="BicycleKin-v0", entry_point="gym_bicycle.envs:BicycleKin")

gym.envs.register(
 id='BicycleKin-v0',
 entry_point='gym_bicycle.envs:BicycleKin'
) 