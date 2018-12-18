import subprocess
import argparse
import shlex

parser = argparse.ArgumentParser(description='Program to Run Carla Servers in Docker')
parser.add_argument('--starting_port', type=int, help='starting port', default='2000')
parser.add_argument('--ids-gpus', type=str, help='string containing the gpu ids', required=True)
parser.add_argument('--image-name', type=str, help='docker image name', default='carla-server')
parser.add_argument('--num-servers', type=int, help='number of servers', default=1)
args = parser.parse_args()

for i in range(args.num_servers):
    gpu_id = args.ids_gpus[i % len(args.ids_gpus)]
    port = args.starting_port + i*3
    cmd = "nvidia-docker run --rm -e NVIDIA_VISIBLE_DEVICES={} -p {}-{}:2000-2002 {} /bin/bash -c \"sed -i '5i sync' ./CarlaUE4.sh; ./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=10 -carla-settings=\"CarlaSettings.ini\"\"".format(gpu_id, port, port+2, args.image_name)
    print i, cmd
    subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
