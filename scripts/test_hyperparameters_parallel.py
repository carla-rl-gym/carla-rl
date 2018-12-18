import os
import subprocess
import argparse
import shlex
import time
import itertools
import yaml
import uuid

# Each hyperparemeter should be a top-level key in the config dict
hyperparams = {'entropy_coef': [0.03],
               'agent': ['ppo'],
               'lr': [7e-5]}
keys, values = zip(*hyperparams.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

parser = argparse.ArgumentParser(description='IntentNetv2 inference')
parser.add_argument('--ids-gpus', type=str, help='string containing the gpu ids', required=True)
parser.add_argument('--starting-port', type=int, help='starting port', default='2000')
parser.add_argument('--video-interval', type=int, help='video interval', default='50')
parser.add_argument('--num-processes', type=int, default=5)
parser.add_argument('--save-dir', type=str, help='directory to save all results', default='./outputs_hyperparam_tuning/')
parser.add_argument('--config', type=str, default='PythonClient/config/base.yaml', help='base config file for client')
parser.add_argument('--server-image-name', type=str, help='docker image name for server', default='carla-server')
parser.add_argument('--client-image-name', type=str, help='docker image name for client', default='carla-client')
parser.add_argument('--clients-only', action='store_true', default=False, help='only run clients')
args = parser.parse_args()

BASE_FOLDER = '.carla-rl-configs'

if not os.path.exists(BASE_FOLDER):
    os.mkdir(BASE_FOLDER)

def create_temp_config(experiment):
    filename = os.path.join(BASE_FOLDER, str(uuid.uuid4()))
    exp_str = ""
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    base_config['num_processes'] = args.num_processes
    for key, value in experiment.iteritems():
        assert key in base_config, 'Invalid value in the experiment'
        base_config[key] = value
        exp_str += "_{}-{}".format(key, value)

    with open(filename, 'w') as f:
        yaml.dump(base_config, f) 

    return filename, exp_str

# Run servers
if not args.clients_only:
    print 'Creating servers' 
    for i, experiment in enumerate(experiments):
        gpu_id = args.ids_gpus[i % len(args.ids_gpus)]
        for j in range(args.num_processes):
            port = args.starting_port + (i * args.num_processes + j) * 3
            cmd = "nvidia-docker run --rm -e NVIDIA_VISIBLE_DEVICES={} -p {}-{}:2000-2002 {} /bin/bash -c \"sed -i '5i sync' ./CarlaUE4.sh; ./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=10 -carla-settings=\"CarlaSettings.ini\"\"".format(gpu_id, port, port+2, args.server_image_name)
            print cmd, "\n\n" 
            subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)
        
    time.sleep(20)

# Run clients
print 'Creating clients'
for i, experiment in enumerate(experiments):
    gpu_id = args.ids_gpus[i % len(args.ids_gpus)]
    port = args.starting_port + (i * args.num_processes) * 3

    # Creates a config.yaml file
    config_file, exp_name = create_temp_config(experiment)
    python_cmd = "python PythonClient/train.py \
                    --config {0} \
                    --starting-port {1} \
                    --video-interval {2} \
                    --save-dir {3} \
                    --experiment-name {4}".format(config_file, port, args.video_interval, args.save_dir, exp_name)
    
    cmd = "nvidia-docker run --rm --network=host -e NVIDIA_VISIBLE_DEVICES={} -v {}:/app {} /bin/bash \
                -c \"{}\"".format(gpu_id, os.environ.get('PWD'), args.client_image_name, python_cmd)
    print cmd, "\n\n"
    subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    time.sleep(10)
