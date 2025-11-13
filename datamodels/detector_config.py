## Functions to automatically generate detector configuration yaml files from the FITS headers
import yaml

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
    required_keys = ['detector_type', 'objects']
    required_subkeys = {
        'objects': ['name', 'class', 'filename_format']
    }
    required_outputs_keys = ['id', 'ext_id', 'ext_slice', 'data_slice']

    def __init__(self, config_path=None, fits_path=None, output_path=None):
        self.config = None

        if config_path is not None:
            self.load_config_from_file(config_path)
        elif fits_path is not None:
            if not (isinstance(fits_path, str) and fits_path.endswith('.fits')):
                raise ValueError(f"Input path should not be None and be a FITS file.")
            else:
                self.generate_config_from_fits(fits_path, output_path)

    def generate_config_from_fits(self, fits_path, output_path=None):
        raise NotImplementedError()

    def load_config_from_file(self, config_path):
        try:
            with open(config_path, 'r') as stream:
                self.config = yaml.load(stream, Loader=yaml.FullLoader)
        except Exception as e:
            raise ValueError(f"Error loading detector configuration file: {e}")
        self.validate_config()

    def validate_config(self):
        for key in self.required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
            else:
                if key in self.required_subkeys.keys():
                    for subkey in self.required_subkeys[key]:
                        for item in self.config[key]:
                            if subkey not in item.keys():
                                raise ValueError(f"Missing required subkey '{subkey}' in config key '{key}'.")

        for item in self.config["objects"]:
            if 'outputs' in item.keys():
                for output in item['outputs']:
                    for outkey in self.required_outputs_keys:
                        if outkey not in output.keys():
                            raise ValueError(f"Missing required output subkey '{outkey}' in object '{item['name']}'.")



    def __call__(self, *args, **kwargs):
        return self.config
