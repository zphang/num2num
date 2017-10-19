from num2num.config import Configuration
from num2num.orchestrate import run_sample


if __name__ == "__main__":
    config_ = Configuration.parse_configuration()
    run_sample(config_)
