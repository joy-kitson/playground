'''Entry point into the agents module set'''
from .base_agent import BaseAgent
from .docker_agent import DockerAgent
from .http_agent import HttpAgent
from .player_agent import PlayerAgent
from .player_agent_blocking import PlayerAgentBlocking
from .random_agent import RandomAgent
from .simple_agent import SimpleAgent
from .tensorforce_agent import TensorForceAgent
from .juke_bot import JukeBot
from .juke_radio import JukeRadio
from .juke_bot_deep import JukeBotDeep
from .juke_bot_deep_conv import JukeBotDeepConv