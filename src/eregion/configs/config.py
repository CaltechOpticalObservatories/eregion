from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
import os
import yaml
from eregion.utils import configure_logger, slice_constructor

yaml.add_constructor('!slice', slice_constructor)

########################################### Generic Config Loader Base Class ########################################
class ConfigLoader(ABC):
    required_keys = []

    def __init__(self, config_input, runtime_variables: Mapping | None = None, enable_env_vars: bool = False):
        """
        Initialize the ConfigLoader either from a YAML config file path, config data in string or dict format.
        :param config_input: str or dict
            Path to a YAML config file or config data as a string or dictionary.
        """
        self.config = None
        self.logger = configure_logger(self.__class__.__name__)
        self.runtime_variables = dict(runtime_variables or {})
        self.enable_env_vars = enable_env_vars

        match config_input:
            case str():
                if config_input.endswith(('.yaml', '.yml')):
                    self.set_from_file(config_input)
                else:
                    self.set_from_string(config_input)
            case dict():
                self.set_from_dict(config_input)
            case _:
                raise ValueError("config_input must be either a file path, string or a dictionary.")

    def set_from_file(self, config_input: str):
        try:
            with open(config_input, 'r') as stream:
                self.load_config(stream)
            self.validate_config()
            self.logger.info("Config loaded from file '{}'".format(config_input))
        except Exception as e:
            raise ValueError(f"Error reading config file {config_input}: {e}")

    def set_from_string(self, config_input: str):
        try:
            self.load_config(config_input)
            self.validate_config()
            self.logger.info("Config loaded from string input")
        except Exception as e:
            raise ValueError(f"Error parsing input string into yaml: {e}")

    def set_from_dict(self, config_input: dict):
        self.config = deepcopy(config_input)
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
                raise KeyError(f"Missing required config key: {key}")

        try:
            resolver = ConfigVariableResolver(runtime_variables=self.runtime_variables,
                                              enable_env_vars=self.enable_env_vars)
            self.config = resolver.resolve(deepcopy(self.config))
        except InterpolationError as e:
            raise ValueError(f"Error resolving interpolations in configuration: {e}")

### Detector Configuration Class ###
class DetectorConfig(ConfigLoader):
    required_keys = ['detector_type', 'detector_output_class', 'objects']
    required_objects_keys = ['name', 'class', 'properties', 'outputs']
    required_properties_keys = ['x_size', 'y_size', 'pixel_size']

    def __init__(self, config_input, runtime_variables: Mapping | None = None, enable_env_vars: bool = False):
        """
        Initialize the DetectorConfig either from a YAML config file path, config data in string or json format,
        or generate from a FITS file.
        :param config_input: str or dict
            Path to a YAML config file or config data as a string or dictionary.
        """
        super().__init__(config_input, runtime_variables=runtime_variables, enable_env_vars=enable_env_vars)

    def validate_config(self):
        for key in self.required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required config key: {key}")

        for obj in self.config['objects']:
            for key in self.required_objects_keys:
                if key not in obj:
                    raise KeyError(f"Missing required object key: {key} in object {obj.get('name', 'unknown')}")

            for prop_key in self.required_properties_keys:
                if prop_key not in obj['properties']:
                    raise KeyError(f"Missing required property key: {prop_key} in object {obj.get('name', 'unknown')}")

        super().validate_config()

### Pipeline Configuration Class ###
class PipelineConfig(ConfigLoader):
    required_keys = ['pipelines']
    required_pipeline_keys = ['name', 'lazy', 'nodes']

    def __init__(self, config_input, runtime_variables: Mapping | None = None, enable_env_vars: bool = False):
        """
        Initialize the PipelineConfig either from a YAML config file path or config data in string or dict format.
        :param config_input: str or dict
            Path to a YAML config file or config data as a string or dictionary.
        """
        super().__init__(config_input, runtime_variables=runtime_variables, enable_env_vars=enable_env_vars)

    def validate_config(self):
        for key in self.required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required config key: {key}")

        for pipeline in self.config['pipelines']:
            for key in self.required_pipeline_keys:
                if key not in pipeline:
                    raise KeyError(f"Missing required pipeline key: {key} in pipeline {pipeline.get('name', 'unknown')}")

            if pipeline['lazy']:
                assert 'source' in pipeline, f"Missing required key 'source' for lazy pipeline {pipeline.get('name', 'unknown')}"

        super().validate_config()


#################### Config variable resolver for interpolation ####################
_ESCAPED_INTERPOLATION = "\u0000REGION_ESCAPED_INTERPOLATION\u0000"

class InterpolationError(ValueError):
    pass

class ConfigVariableResolver:
    def __init__(self, runtime_variables: Mapping | None = None, enable_env_vars: bool = False):
        self.runtime_variables = dict(runtime_variables or {})
        self.enable_env_vars = enable_env_vars
        self._stack: list[str] = []

    def resolve(self, value, current_path: str = "config"):
        if isinstance(value, dict):
            return {key: self.resolve(val, f"{current_path}.{key}") for key, val in value.items()}
        if isinstance(value, list):
            return [self.resolve(item, f"{current_path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, str):
            return self._resolve_string(value, current_path)
        return value

    def _resolve_string(self, text: str, current_path: str):
        if not self._is_var(text):
            return text

        working = text.replace(r"\${", _ESCAPED_INTERPOLATION)
        parts: list[tuple[str, str]] = []
        index = 0
        while index < len(working):
            start = working.find("${", index)
            if start == -1:
                parts.append(("text", working[index:]))
                break

            if start > index:
                parts.append(("text", working[index:start]))

            end = working.find("}", start + 2)
            if end == -1:
                raise InterpolationError(
                    f"Malformed interpolation token at {current_path}: missing '}}' in {text!r}"
                )

            token = working[start + 2 : end]
            if not token:
                raise InterpolationError(f"Malformed interpolation token at {current_path}: empty placeholder in {text!r}")
            if "${" in token or "}" in token:
                raise InterpolationError(
                    f"Malformed interpolation token at {current_path}: nested placeholder in {text!r}"
                )
            parts.append(("placeholder", token))
            index = end + 1

        if len(parts) == 1 and parts[0][0] == "placeholder":
            return self._resolve_placeholder(parts[0][1], current_path)

        rendered = []
        for kind, piece in parts:
            if kind == "text":
                rendered.append(piece.replace(r"\${", "${").replace(_ESCAPED_INTERPOLATION, "${"))
            else:
                rendered.append(str(self._resolve_placeholder(piece, current_path)))
        return "".join(rendered)

    def _resolve_placeholder(self, token: str, current_path: str):
        name, default = self._split_token(token, current_path)
        return self._resolve_reference(name, default, current_path)

    @staticmethod
    def _is_var(text: str) -> bool:
        return "${" in text or r"\${" in text

    @staticmethod
    def _split_token(token: str, current_path: str):
        name, sep, default = token.partition(":")
        if not name or not name.strip():
            raise InterpolationError(f"Malformed interpolation token at {current_path}: missing variable name in {token!r}")
        if name != name.strip() or any(ch in name for ch in " {}\t\r\n{}"):
            raise InterpolationError(f"Malformed interpolation token at {current_path}: invalid variable name in {token!r}")
        if sep and default == "":
            raise InterpolationError(f"Malformed interpolation token at {current_path}: empty default in {token!r}")
        return name, default if sep else None

    def _resolve_reference(self, name: str, default: str | None, current_path: str):
        if name in self._stack:
            cycle = " -> ".join(self._stack + [name])
            raise InterpolationError(f"Interpolation cycle detected at {current_path}: {cycle}")

        self._stack.append(name)
        try:
            found, value = self._lookup(name)
            if found:
                return self.resolve(value, f"var:{name}")
            if default is not None:
                return self._resolve_default(default, f"var:{name}")
            raise InterpolationError(f"Unknown interpolation variable '{name}' at {current_path}")
        finally:
            self._stack.pop()

    def _lookup(self, name: str):
        if name in self.runtime_variables:
            return True, self.runtime_variables[name]

        if self.enable_env_vars and name in os.environ:
            return True, os.environ[name]

        if "." in name:
            current = self.runtime_variables
            for segment in name.split("."):
                if not isinstance(current, Mapping) or segment not in current:
                    break
                current = current[segment]
            else:
                return True, current

        return False, None

    def _resolve_default(self, default: str, current_path: str):
        resolved = self._resolve_string(default, current_path) if self._is_var(default) else default
        if isinstance(resolved, str) and not self._is_var(default):
            return yaml.safe_load(resolved)
        return resolved