## Functions to automatically generate detector configuration yaml files from the FITS headers
import yaml
import logging
import os

logger = logging.getLogger(__name__)

# Create a yaml constructor for slice objects
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

class DetectorConfig:
    required_keys = ['detector_type', 'detector_output_class', 'objects']
    required_objects_keys = ['name', 'class', 'properties', 'outputs']
    required_properties_keys = ['x_size', 'y_size', 'pixel_size']

    def __init__(self, config_input=None, fits_path=None, output_path=None):
        """
        Initialize the DetectorConfig either from a YAML config file path, config data in string or json format,
        or generate from a FITS file.
        :param config_input: str or dict, optional
            Path to a YAML config file or config data as a string or dictionary.
        :param fits_path: str, optional
            Path to a FITS file to generate configuration from.
        :param output_path: str, optional
            Path to save the generated configuration file if generating from FITS.
        """
        self.config = None

        if config_input is not None:
            if os.path.isfile(config_input):
                with open(config_input, 'r') as stream:
                    self.load_config(stream)
            elif isinstance(config_input, str):
                try:
                    self.load_config(config_input)
                except Exception as e:
                    raise ValueError(f"Error parsing config_input string: {e}")
            elif isinstance(config_input, dict):
                self.config = config_input
            else:
                raise ValueError("config_input must be either a file path, string or a dictionary.")
        elif config_input is None and fits_path is not None:
            if not (isinstance(fits_path, str) and '.fits' in fits_path):
                raise ValueError(f"Input path should not be None and be a FITS file.")
            else:
                self.generate_config_from_fits(fits_path, output_path)
        else:
            logger.warning("No configuration, nor FITS file to generate from. Initializing empty config.")
            self.config = {}

    def load_config(self, stream):
        try:
            self.config = yaml.load(stream, Loader=yaml.FullLoader)
        except Exception as e:
            raise ValueError(f"Error loading detector configuration file: {e}")
        self.validate_config()

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

    def generate_config_from_fits(self, fits_path, output_path=None):
        raise NotImplementedError()


