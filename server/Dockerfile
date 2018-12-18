FROM carlasim/carla:0.8.2

USER root 

RUN sudo apt-get update && sudo apt-get install -y vim

# COPY OVER SETTINGS
COPY --chown=carla ./CarlaSettings.ini .

USER carla

