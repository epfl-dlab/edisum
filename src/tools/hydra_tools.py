import hydra


def get_config(config_name, configs_folder="../../configs", overrides=[], work_dir=".", data_dir="./data"):
    default_overrides = [f"work_dir={work_dir}", f"data_dir={data_dir}"]
    overrides += default_overrides

    with hydra.initialize(version_base="1.2", config_path=configs_folder):
        config = hydra.compose(config_name=config_name, overrides=overrides)

    return config