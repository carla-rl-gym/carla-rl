import numpy as np
import cv2
import gym

class CameraException(Exception):

    def __init__(self, id):
        super(CameraException, self).__init__()
        self.id = id


class CarlaObservationConverter(object):

    def __init__(self, h=84, w=84, rel_coord_system=False):

        self.c = 3
        self.h = h
        self.w = w
        self.rel_coord_system = rel_coord_system
        if self.rel_coord_system:
            self.vbounds = np.array([
                [0, 30],
                [-100, 100],
                [-100, 100],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 100],
                [0, 100],
                [0, 1],
                [0, 1]
            ])

            self.world_pos_bounds = np.array([
                [0, 100],
                [0, 100],
                [0, 1]
            ])

        else:
            self.vbounds = np.array([
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 30],
                [-100, 100],
                [-100, 100],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 100]  # TODO: Fine-tune
            ])

        # From Carla Code - Deprecated/PythonClient/carla/planner/planner.py
        self.REACH_GOAL = 0.0
        self.GO_STRAIGHT = 5.0
        self.TURN_RIGHT = 4.0
        self.TURN_LEFT = 3.0
        self.LANE_FOLLOW = 2.0

    def direction_to_string(self, directions):
        if float(np.isclose(directions, self.REACH_GOAL)):
            return 'REACH_GOAL'
        if float(np.isclose(directions, self.GO_STRAIGHT)):
            return 'GO_STRAIGHT'
        if float(np.isclose(directions, self.TURN_RIGHT)):
            return 'TURN_RIGHT'
        if float(np.isclose(directions, self.TURN_LEFT)):
            return 'TURN_LEFT'
        if float(np.isclose(directions, self.LANE_FOLLOW)):
            return 'LANE_FOLLOW'
        assert False, 'Unknown direction'

    def convert(self, measurements, sensor_data, directions, target, env_id):

        player_measurements = measurements.player_measurements

        if self.rel_coord_system:
            target_rel_x, target_rel_y = self.get_relative_location_target(
                                        player_measurements.transform.location.x,
                                        player_measurements.transform.location.y,
                                        player_measurements.transform.rotation.yaw,
                                        target.location.x,
                                        target.location.y,)
            target_rel_norm = np.linalg.norm(np.array([target_rel_x, target_rel_y]))
            target_rel_x_unit = target_rel_x / target_rel_norm
            target_rel_y_unit = target_rel_y / target_rel_norm

            v = np.array([
                player_measurements.forward_speed,
                player_measurements.acceleration.x,
                player_measurements.acceleration.y,
                float(np.isclose(directions, self.REACH_GOAL)),
                float(np.isclose(directions, self.GO_STRAIGHT)),
                float(np.isclose(directions, self.TURN_RIGHT)),
                float(np.isclose(directions, self.TURN_LEFT)),
                float(np.isclose(directions, self.LANE_FOLLOW)),
                target_rel_x,
                target_rel_y,
                target_rel_x_unit,
                target_rel_y_unit
            ])

            world_pos = np.array([
                player_measurements.transform.location.x,
                player_measurements.transform.location.y,
                player_measurements.transform.rotation.yaw
            ])

        else:
            v = np.array([
                player_measurements.transform.location.x,
                player_measurements.transform.location.y,
                player_measurements.transform.rotation.yaw,
                player_measurements.forward_speed,
                player_measurements.acceleration.x,
                player_measurements.acceleration.y,
                float(np.isclose(directions, self.REACH_GOAL)),
                float(np.isclose(directions, self.GO_STRAIGHT)),
                float(np.isclose(directions, self.TURN_RIGHT)),
                float(np.isclose(directions, self.TURN_LEFT)),
                float(np.isclose(directions, self.LANE_FOLLOW)),
                player_measurements.collision_vehicles,
                player_measurements.collision_pedestrians,
                player_measurements.collision_other,
                player_measurements.intersection_otherlane,
                player_measurements.intersection_offroad,
                target.location.x,
                target.location.y,
                target.rotation.yaw
            ])

        try:
            img = cv2.resize(sensor_data['CameraRGB'].data, (self.h, self.w)) / 255.0
        except:
            raise CameraException(env_id)
        img = np.transpose(img, (2, 0, 1))

        if self.rel_coord_system:
            return {'img': img, 'v': v, 'world_pos': world_pos}
        else:
            return {'img': img, 'v': v}


    def get_relative_location_target(self, loc_x, loc_y, loc_yaw, target_x, target_y):

        veh_yaw = loc_yaw * np.pi / 180
        veh_dir_world = np.array([np.cos(veh_yaw), np.sin(veh_yaw)])
        veh_loc_world = np.array([loc_x, loc_y])
        target_loc_world = np.array([target_x, target_y])
        d_world = target_loc_world - veh_loc_world
        dot = np.dot(veh_dir_world, d_world)
        det = veh_dir_world[0]*d_world[1] - d_world[0]*veh_dir_world[1]
        rel_angle = np.arctan2(det, dot)
        target_location_rel_x = np.linalg.norm(d_world) * np.cos(rel_angle)
        target_location_rel_y = np.linalg.norm(d_world) * np.sin(rel_angle)

        return target_location_rel_x.item(), target_location_rel_y.item()


    def get_observation_space(self):

        img_shape = (self.c, self.h, self.w)
        img_box = gym.spaces.Box(low=0, high=1, shape=img_shape, dtype=np.float32)
        v_low = self.vbounds[:, 0]
        v_high = self.vbounds[:, 1]
        v_box = gym.spaces.Box(low=v_low, high=v_high, dtype=np.float32)
        if self.rel_coord_system:
            world_pos_low = self.world_pos_bounds[:, 0]
            world_pos_high = self.world_pos_bounds[:, 1]
            world_pos_box = gym.spaces.Box(low=world_pos_low, high=world_pos_high, dtype=np.float32)
            d = {'img': img_box, 'v': v_box, 'world_pos': world_pos_box}
        else:
            d = {'img': img_box, 'v': v_box}
        return gym.spaces.Dict(d)
