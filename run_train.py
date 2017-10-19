from num2num.config import Configuration
from num2num.orchestrate import run_train


if __name__ == "__main__":
    config_ = Configuration.parse_configuration()
    run_train(config_)
