from carla.driving_benchmark.experiment_suites.corl_2017 import CoRL2017
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.driving_benchmark.experiment import Experiment


class TrainingSuite(CoRL2017):

    def __init__(self, city_name, subset=None):
        
        self._subset = subset
        super(TrainingSuite, self).__init__(city_name)


    def build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.
        """

        # We set the camera
        # This single RGB camera is used on every experiment

        camera = Camera('CameraRGB')
        camera.set(FOV=100)
        camera.set_image_size(800, 600)
        camera.set_position(2.0, 0.0, 1.4)
        camera.set_rotation(-15.0, 0, 0)

        if self._city_name == 'Town01':
            
            if self._subset == 'keep_lane':
                poses_tasks = [self._poses_town01()[0]]
                vehicles_tasks = [0]
                pedestrians_tasks = [0]

            elif self._subset == 'one_turn':
                poses_tasks = [self._poses_town01()[1]]
                vehicles_tasks = [0]
                pedestrians_tasks = [0]

            elif self._subset == 'keep_lane_one_turn':
                poses_tasks = self._poses_town01()[:2]
                vehicles_tasks = [0, 0]
                pedestrians_tasks = [0, 0]

            elif self._subset == 'no_dynamic_objects':
                poses_tasks = self._poses_town01()[:3]
                vehicles_tasks = [0, 0, 0]
                pedestrians_tasks = [0, 0, 0]

            elif self._subset is None:
                poses_tasks = self._poses_town01()
                vehicles_tasks = [0, 0, 0, 20]
                pedestrians_tasks = [0, 0, 0, 50]

            else:
                raise ValueError("experiments-subset must be keep_lane or keep_lane_one_turn or no_dynamic_objects or None")
        
        else:
            raise ValueError("city must be Town01 for training")

        experiments_vector = []

        for i, iteration in enumerate(range(len(poses_tasks))):

            for weather in self.weathers:
            
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather
                )
                # Add all the cameras that were set for this experiments

                conditions.add_sensor(camera)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                )
                
                experiments_vector.append(experiment)

        return experiments_vector