import yaml

def class_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {k: class_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    else:
        return obj

class Args:
    def __init__(self, dicts):
        for key, value in dicts.items():
            if isinstance(value, dict):
                value = Args(value)
            setattr(self, key, value)

    def to_dict(self):
        result = {}
        for key in self.__dict__:
            value = getattr(self, key)
            if isinstance(value, Args):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_data = f.read()
    data = yaml.safe_load(yaml_data)
    return data

def merge_configs(dict1, dict2):
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        for key, value in dict2.items():
            if key not in dict1.keys():
                dict1.update({
                    key: value
                })
            else:
                dict1[key] = merge_configs(dict1[key], value)

    return dict1

def parse(config_path, args=None):  # config_path = 'configs/xxx.yaml'
    seen_paths = set()
    data = {} if args is None else class_to_dict(args)
    current_path = config_path

    while current_path:
        if current_path in seen_paths:
            raise ValueError(f"Circular inheritance detected for config: {current_path}")

        seen_paths.add(current_path)
        current_data = load_configs(current_path)

        data = merge_configs(data, current_data)
        current_path = current_data.get('parent_config', None)

    config_args = Args(data)
    return config_args
