# Carla RL Project

## Installation and Setup

### Running the CARLA Server
Our program uses the CARLA simulator as the environment. The easiest way to install CARLA is to use the Docker container by running,
`docker pull carlasim/carla:0.8.2`

We change the default settings (the timeout) when running the server and therefore we have our own `carla-server` docker image that builds on this image. You can build it by running
`docker build server -t carla-server`.

Next, you can run the server Docker container with
`nvidia-docker run --rm -it -p 2000-2002:2000-2002 carlasim/carla:0.8.2 /bin/bash`

Note that this requires `nvidia-docker` to be installed on your machine (which means you will also need a GPU). Finally, inside the docker container you can
run the server with
`./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600 `

However, since we often require running more than 1 server, we recommend using the script `server/run_servers.py` to run the CARLA server. You can run N servers by running (the logs for stdout and stderr will be under `server_output` folder):
`python server/run_servers.py  --num-servers N`
In order to see the servers output `docker logs -ft CONTAINER_ID` follows and tails it.

### Running the client (training code, benchmark code)
Our code requires:
* Python 3
* PyTorch
* OpenAI Gym
* OpenAI Baselines

We suggest that you use our own Dockerfile to install all these dependencies. You can build our client Dockerfile with,
`docker build client -t carla-client`
then you can run
`nvidia-docker run -it --network=host -v $PWD:/app carla-client /bin/bash`
The `--network=host` flag allows the Docker container to make requests to the server. Once you are inside
the container, you can run any of our scripts like `python client/train.py --config client/config/base.yaml`.

### Arguments and Config Files
Our `client/train.py` script uses both arguments and a configuration file. The configuration file specifies all components of the model. The config file should have everything necessary to reproduce the results of a given model. The arguments of the script deal with things that are independent of the model (this includes things, like for example, how often to create videos or log to Tensorboard)


### Hyperparameter Tuning
To test a set of hyperparemeters see the `scripts/test_hyperparameters_parallel.py` script. This will let you specify a set of hyperparameters to test different from those specified in the `client/config/base.yaml` file.

## Benchmark Results

### A2C
To reproduce our results, run a CARLA server and inside the `carla-client` docker run,
`python client/train.py --config client/config/a2c.yaml`

### ACKTR
To reproduce our results, run a CARLA server and inside the `carla-client` docker run,
`python client/train.py --config client/config/acktr.yaml`

### PPO
To reproduce our results, run a CARLA server and inside the `carla-client` docker run,
`python client/train.py --config client/config/ppo.yaml`

### On-Policy HER
To reproduce our results, run a CARLA server and inside the `carla-client` docker run,
`python client/train.py --config client/config/her.yaml`
