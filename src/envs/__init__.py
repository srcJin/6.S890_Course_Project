import os
import sys

from .multiagentenv import MultiAgentEnv

# Import wrappers only when needed to avoid dependency issues
def _import_gymma():
    from .gymma import GymmaWrapper
    return GymmaWrapper

def _import_simcity():
    from .simcity_wrapper import SimCityWrapper
    return SimCityWrapper

def _import_simcity_scale_up():
    from .simcity_scale_up_wrapper import SimCityScaleUpWrapper
    return SimCityScaleUpWrapper


# if sys.platform == "linux":
#     os.environ.setdefault(
#         "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
#     )


# def __check_and_prepare_smac_kwargs(kwargs):
#     assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
#     assert kwargs[
#         "common_reward"
#     ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
#     del kwargs["common_reward"]
#     del kwargs["reward_scalarisation"]
#     assert "map_name" in kwargs, "Please specify the map_name in the env_args"
#     return kwargs


# def smaclite_fn(**kwargs) -> MultiAgentEnv:
#     kwargs = __check_and_prepare_smac_kwargs(kwargs)
#     return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    GymmaWrapper = _import_gymma()
    return GymmaWrapper(**kwargs)


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
# REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn
REGISTRY["simcity"] = lambda **kwargs: env_fn(_import_simcity(), **kwargs)
REGISTRY["simcity_scale_up"] = lambda **kwargs: env_fn(_import_simcity_scale_up(), **kwargs)

# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
# def register_smac():
#     from .smac_wrapper import SMACWrapper

#     def smac_fn(**kwargs) -> MultiAgentEnv:
#         kwargs = __check_and_prepare_smac_kwargs(kwargs)
#         return SMACWrapper(**kwargs)

#     REGISTRY["sc2"] = smac_fn


# def register_smacv2():
#     from .smacv2_wrapper import SMACv2Wrapper

#     def smacv2_fn(**kwargs) -> MultiAgentEnv:
#         kwargs = __check_and_prepare_smac_kwargs(kwargs)
#         return SMACv2Wrapper(**kwargs)

#     REGISTRY["sc2v2"] = smacv2_fn
