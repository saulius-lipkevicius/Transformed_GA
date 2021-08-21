import ruamel.yaml


config_name = 'config.yaml'

with open(config_name, 'r') as stream:
    try:
        yaml = ruamel.yaml.YAML()
        config = yaml.load(stream)
        print(config["Datasets"]['val'])
    except yaml.YAMLError as exc:
        print(exc)
config["Datasets"]['val'] = './heheheh/val.csv'
#  updating
print(config)
with open('config.yaml', 'w') as conf:
    yaml.dump(config, conf)
