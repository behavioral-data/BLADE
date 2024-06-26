import asyncio
import os.path as osp
import time

import hydra
from hydra.core.hydra_config import HydraConf, HydraConfig
from omegaconf import DictConfig, OmegaConf

from blade_bench.baselines.config import BenchmarkConfig
from blade_bench.baselines.main import RunLLMAndEval, run_main
from blade_bench.data.datamodel.transforms import (
    TransformDataReturn,
)  # ❗️ this import needs to be kept here
from blade_bench.logger import logger, formatter


@hydra.main(
    version_base=None,
    config_path=("conf"),
    config_name="run_baseline",
)
def my_app(cfg: DictConfig) -> None:
    # cfg = OmegaConf.to_yaml(cfg)
    hydra_cfg: HydraConf = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    OmegaConf.resolve(cfg)
    cfg_agent = dict(cfg)
    cfg_agent["output_dir"] = output_dir
    agent_config = BenchmarkConfig(**cfg_agent)
    log_file = f"{hydra_cfg.help.app_name}.log"
    log_file_jsonl = f"{hydra_cfg.help.app_name}.jsonl"
    i1 = logger.add(osp.join(output_dir, log_file), format=formatter.format)
    i2 = logger.add(osp.join(output_dir, log_file_jsonl), format="{extra[serialized]}")
    agent = RunLLMAndEval(agent_config)
    asyncio.run(run_main(agent, hydra_cfg.mode, cfg_agent.get("run_name", "default")))
    logger.remove(i1)
    logger.remove(i2)


if __name__ == "__main__":
    my_app()
