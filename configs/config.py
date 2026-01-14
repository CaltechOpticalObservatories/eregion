from abc import ABC, abstractmethod
import yaml
import logging
import os

# A yaml constructor for slice objects
def slice_constructor(loader, node):
    values = loader.construct_sequence(node)
    # slice will be created from a list, e.g., [start, stop, step]
    start, stop, step = None, None, None
    if len(values) == 1:
        stop = values[0]
    elif len(values) == 2:
        start, stop = values
    elif len(values) == 3:
        start, stop, step = values
    else:
        raise ValueError("Invalid number of arguments for slice.")
    return slice(start, stop, step)

yaml.add_constructor('!slice', slice_constructor)


### Generic Config Loader Base Class ###
class ConfigLoader(ABC):
    required_keys = []

    def __init__(self, config_input):
        """
        Initialize the ConfigLoader either from a YAML config file path, config data in string or dict format.
        :param config_input: str or dict
            Path to a YAML config file or config data as a string or dictionary.
        """
        self.config = None
        self.logger = logging.getLogger(__name__)

        if isinstance(config_input, str) and os.path.isfile(config_input) and config_input.endswith(('.yaml', '.yml')):
            self.set_from_file(config_input)
        elif isinstance(config_input, str) and not os.path.isfile(config_input):
            self.set_from_string(config_input)
        elif isinstance(config_input, dict):
            self.set_from_dict(config_input)
        else:
            raise ValueError("config_input must be either a file path, string or a dictionary.")

    def set_from_file(self, input: str):
        try:
            with open(input, 'r') as stream:
                self.load_config(stream)
            self.logger.info("Config loaded from file '{}'".format(input))
        except Exception as e:
            raise ValueError(f"Error reading config file {input}: {e}")
        self.validate_config()

    def set_from_string(self, input: str):
        try:
            self.load_config(input)
            self.logger.info("Config loaded from string input")
        except Exception as e:
            raise ValueError(f"Error parsing input string into yaml: {e}")
        self.validate_config()

    def set_from_dict(self, input: dict):
        self.config = input
        self.validate_config()
        self.logger.info("Config loaded from dict input")

    def load_config(self, stream):
        try:
            self.config = yaml.load(stream, Loader=yaml.FullLoader)
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    @abstractmethod
    def validate_config(self):
        for key in self.required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")


### Detector Configuration Class ###
class DetectorConfig(ConfigLoader):
    required_keys = ['detector_type', 'detector_output_class', 'objects']
    required_objects_keys = ['name', 'class', 'properties', 'outputs']
    required_properties_keys = ['x_size', 'y_size', 'pixel_size']

    def __init__(self, config_input):
        """
        Initialize the DetectorConfig either from a YAML config file path, config data in string or json format,
        or generate from a FITS file.
        :param config_input: str or dict
            Path to a YAML config file or config data as a string or dictionary.
        """
        super().__init__(config_input)

    def validate_config(self):
        for key in self.required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        for obj in self.config['objects']:
            for key in self.required_objects_keys:
                if key not in obj:
                    raise ValueError(f"Missing required object key: {key} in object {obj.get('name', 'unknown')}")

            for prop_key in self.required_properties_keys:
                if prop_key not in obj['properties']:
                    raise ValueError(f"Missing required property key: {prop_key} in object {obj.get('name', 'unknown')}")



### Pipeline Configuration Class ###
class PipelineConfig(ConfigLoader):
    required_keys = ['pipeline']

    def __init__(self, config_input):
        """
        Initialize the PipelineConfig either from a YAML config file path or config data in string or dict format.
        :param config_input: str or dict
            Path to a YAML config file or config data as a string or dictionary.
        """
        super().__init__(config_input)

    def validate_config(self):
        for key in self.required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
