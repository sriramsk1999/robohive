# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/robohive
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """
DESC = """
TUTORIAL: Arm+Gripper tele-op using input devices (keyboard / spacenav) \n
    - NOTE: Tutorial is written for franka arm and robotiq gripper. This demo is a tutorial, not a generic functionality for any any environment
EXAMPLE:\n
    - python tutorials/ee_teleop.py -e rpFrankaRobotiqData-v0\n
"""
# TODO: (1) Enforce pos/rot/grip limits (b) move gripper to delta commands

from robohive.utils.quat_math import euler2quat, mat2quat, mulQuat, quat2mat, euler2mat
from robohive.utils.inverse_kinematics import IKResult, qpos_from_site_pose
from robohive.logger.roboset_logger import RoboSet_Trace
from robohive.logger.grouped_datasets import Trace as RoboHive_Trace
import numpy as np
import click
import gym
import os

try:
    from vtils.input.keyboard import KeyInput as KeyBoard
    from vtils.input.spacemouse import SpaceMouse
except ImportError as e:
    raise ImportError("Please install vtils -- https://github.com/vikashplus/vtils")
import open3d as o3d
import pickle
import cv2

def load_hoi4d_trajectory(base_path):
    """loads a test trajectory from a hardcoded path.
    The minimum requirement is the hand pose in the
    camera frame (cam2hand) and the camera pose in the world (world2cam).
    """
    cam_trajectory = o3d.io.read_pinhole_camera_trajectory(f"{base_path}/3Dseg/output.log")
    K = cam_trajectory.parameters[0].intrinsic.intrinsic_matrix
    cam2camWorld = np.array([i.extrinsic for i in cam_trajectory.parameters])
    camWorld2cam = np.linalg.inv(cam2camWorld)

    cam2hand = []
    # in some frames, the hand might not be visible, filter these out
    idxs = np.array(sorted([int(i.split('.')[0]) for i in os.listdir(f"{base_path}/handpose")]))
    for idx in idxs:
        f = open(f"{base_path}/handpose/{idx}.pickle", "rb")
        data = pickle.load(f)
        pose = np.eye(4)
        global_rot = data['poseCoeff'][:3]
        global_rot = cv2.Rodrigues(global_rot)[0]
        pose[:3,:3] = global_rot
        pose[:3, 3] = data['trans']
        cam2hand.append(pose)
        f.close()
    cam2hand = np.array(cam2hand)

    camWorld2hand = camWorld2cam[idxs] @ cam2hand
    return camWorld2hand


def vis_trajectory(traj):
    """
    A utility function to visualize trajectory
    """
    frames = []
    trajectory_points = []
    for pose in traj:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        frame.transform(pose)
        frames.append(frame)
        # Extract the translation (position) of each pose for the trajectory line
        trajectory_points.append(pose[:3, 3])

    # Create a LineSet to visualize the trajectory as a line between points
    trajectory_points = np.array(trajectory_points)
    lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]  # Connect points with lines
    colors = [[0, 1, 0] for _ in lines]  # Green color for the lines

    # Create the LineSet object
    trajectory_line = o3d.geometry.LineSet()
    trajectory_line.points = o3d.utility.Vector3dVector(trajectory_points)
    trajectory_line.lines = o3d.utility.Vector2iVector(lines)
    trajectory_line.colors = o3d.utility.Vector3dVector(colors)

    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    start_sphere.paint_uniform_color([1, 0, 0])  # Red for start
    start_sphere.translate(trajectory_points[0])

    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    end_sphere.paint_uniform_color([0, 0, 1])  # Blue for end
    end_sphere.translate(trajectory_points[-1])

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Visualize everything
    o3d.visualization.draw_geometries(frames + [trajectory_line, start_sphere, end_sphere, origin_frame])

def retarget_hand_trajectory(camWorld2hand, robotWorld2ee):
    """
    Align hand coordinates with end effector.
    retarget the 4x4 extrinsics to quaternion/translation/gripper state
    """
    scale_factor = 0.6
    # TODO: Scale translation more intelligently
    # Scale down the translation of the trajectory
    # so that end effector does not go out of reach of the robot
    camWorld2hand[:, :3, 3] *= scale_factor

    robotWorld2camWorld = robotWorld2ee @ np.linalg.inv(camWorld2hand[0])
    robotWorld2hand = robotWorld2camWorld @ camWorld2hand

    # TODO: Handle rotation properly. Right now its being ignored
    # Set end effector orientation to identity
    robotWorld2hand[:, :3, :3] = np.eye(3)
    # TODO: Validate align_rotation. Is it always necessary?
    # A rotation to orient the end effector correctly, flipped upside down without this
    align_rotation = euler2mat((0, np.pi, 0))
    robotWorld2hand[:, :3, :3] = robotWorld2hand[:, :3, :3] @ align_rotation

    robot_trajectory_quat = mat2quat(robotWorld2hand[:, :3, :3])
    robot_trajectory_pos = robotWorld2hand[:, :3, 3]
    # TODO: Handle gripper state properly
    robot_gripper_state = np.zeros((robotWorld2hand.shape[0], 1))
    trajectory = np.concatenate([robot_trajectory_pos, robot_trajectory_quat, robot_gripper_state], axis=-1)
    return trajectory

# Poll and process keyboard values
def poll_keyboard(input_device):
    # get sensors
    sen = input_device.get_sensors()

    # exit request
    done = True if sen=='esc' else False

    # gripper
    if sen == 'h':
        delta_gripper = -1
    elif sen == 'j':
        delta_gripper = 1
    else:
        delta_gripper = 0

    # positions
    delta_pos = np.array([0, 0, 0])
    if sen == 'g':
        delta_pos[0] = 1
    elif sen == 'f':
        delta_pos[0] = -1
    elif sen == 'r':
        delta_pos[1] = 1
    elif sen == 'v':
        delta_pos[1] = -1
    elif sen == 't':
        delta_pos[2] = 1
    elif sen == 'b':
        delta_pos[2] = -1

    # rotations
    delta_euler = np.array([0, 0, 0])
    if sen == 'up':
        delta_euler[0] = -1
    elif sen == 'down':
        delta_euler[0] = 1
    elif sen == 'left':
        delta_euler[1] = -1
    elif sen == 'right':
        delta_euler[1] = 1
    elif sen == ',':
        delta_euler[2] = -1
    elif sen == '.':
        delta_euler[2] = 1

    return delta_pos, delta_euler, delta_gripper, done


# Poll and process spacemouse values
def poll_spacemouse(input_device):
    # get sensors
    sen = input_device.get_sensors()

    # exit request
    done = True if (sen['left'] and sen['right']) else False

    # gripper
    if sen['left'] == True:
        delta_gripper = -1
    elif sen['right'] == True:
        delta_gripper = 1
    else:
        delta_gripper = 0

    # positions
    delta_pos = np.array([sen['x'], sen['y'], sen['z']])

    # rotations
    delta_euler = np.array([sen['roll'], sen['pitch'], sen['yaw']])

    return delta_pos, delta_euler, delta_gripper, done


@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', default='rpFrankaRobotiqData-v0')
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('-rn', '--reset_noise', type=float, default=0.0, help=('Amplitude of noise during reset'))
@click.option('-an', '--action_noise', type=float, default=0.0, help=('Amplitude of action noise during rollout'))
@click.option('-i', '--input_device', type=click.Choice(['keyboard', 'spacemouse']), help='input to use for teleOp', default='keyboard')
@click.option('-o', '--output', type=str, default="teleOp_trace.h5", help=('Output name'))
@click.option('-h', '--horizon', type=int, help='Rollout horizon', default=1000)
@click.option('-n', '--num_rollouts', type=int, help='number of repeats for the rollouts', default=1)
@click.option('-f', '--output_format', type=click.Choice(['RoboHive', 'RoboSet']), help='Data format', default='RoboHive')
@click.option('-c', '--camera', multiple=True, type=str, default=[], help=('list of camera topics for rendering'))
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'onscreen+offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-gs', '--goal_site', type=str, help='Site that updates as goal using inputs', default='ee_target')
@click.option('-ts', '--teleop_site', type=str, help='Site used for teleOp/target for IK', default='end_effector')
@click.option('-ps', '--pos_scale', type=float, default=0.05, help=('position scaling factor'))
@click.option('-rs', '--rot_scale', type=float, default=0.1, help=('rotation scaling factor'))
@click.option('-gs', '--gripper_scale', type=float, default=1, help=('gripper scaling factor'))
@click.option('-vi', '--vendor_id', type=int, default=9583, help=('Spacemouse vendor id'))
@click.option('-pi', '--product_id', type=int, default=50741, help=('Spacemouse product id'))
# @click.option('-tx', '--x_range', type=tuple, default=(-0.5, 0.5), help=('x range'))
# @click.option('-ty', '--y_range', type=tuple, default=(-0.5, 0.5), help=('y range'))
# @click.option('-tz', '--z_range', type=tuple, default=(-0.5, 0.5), help=('z range'))
# @click.option('-rx', '--roll_range', type=tuple, default=(-0.5, 0.5), help=('roll range'))
# @click.option('-ry', '--pitch_range', type=tuple, default=(-0.5, 0.5), help=('pitch range'))
# @click.option('-rz', '--yaw_range', type=tuple, default=(-0.5, 0.5), help=('yaw range'))
# @click.option('-gr', '--gripper_range', type=tuple, default=(0, 1), help=('z range'))
def main(env_name, env_args, reset_noise, action_noise, input_device, output, horizon, num_rollouts, output_format, camera, render, seed, goal_site, teleop_site, pos_scale, rot_scale, gripper_scale, vendor_id, product_id):
    # x_range, y_range, z_range, roll_range, pitch_range, yaw_range, gripper_range

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env.seed(seed)
    env.env.mujoco_render_frames = True if 'onscreen'in render else False
    goal_sid = env.sim.model.site_name2id(goal_site)
    env.sim.model.site_rgba[goal_sid][3] = 0.2 # make visible

    # prep input device
    if input_device=='keyboard':
        input = KeyBoard()
    elif input_device=='spacemouse':
        input = SpaceMouse(vendor_id=vendor_id, product_id=product_id)
        print("Press both keys to stop listening")

    # prep the logger
    if output_format=="RoboHive":
        trace = RoboHive_Trace("TeleOp Trajectories")
    elif output_format=="RoboSet":
        trace = RoboSet_Trace("TeleOp Trajectories")

    env.reset()
    curr_pos = env.sim.model.site_pos[goal_sid]
    curr_quat =  env.sim.model.site_quat[goal_sid]
    ee_pose = np.eye(4)
    ee_pose[:3, :3] = quat2mat(curr_quat)
    ee_pose[:3, 3] = curr_pos
    base_path = "/home/sriram.sk/desktop/hoi4d_vid/_data_sriram_hoi4d_hoi4d_data_ZY20210800003_H3_C14_N42_S207_s05_T2_vid/"
    camWorld2hand = load_hoi4d_trajectory(base_path)
    trajectory = retarget_hand_trajectory(camWorld2hand, ee_pose)

    # Collect rollouts
    for i_rollout in range(num_rollouts):

        # start a new rollout
        print("rollout {} start".format(i_rollout))
        group_key='Trial'+str(i_rollout); trace.create_group(group_key)
        reset_noise = reset_noise*np.random.uniform(low=-1, high=1, size=env.init_qpos.shape)
        env.reset(reset_qpos=env.init_qpos+reset_noise, blocking=True)

        # recover init state
        obs, rwd, done, env_info = env.forward()
        act = np.zeros(env.action_space.shape)
        gripper_state = 0

        # start rolling out
        for i_step in range(horizon+1):

            # poll input device --------------------------------------
            if input_device=='keyboard':
                delta_pos, delta_euler, delta_gripper, exit_request = poll_keyboard(input)
            elif input_device=='spacemouse':
                delta_pos, delta_euler, delta_gripper, exit_request = poll_spacemouse(input)
            if exit_request:
                print("Rollout done. ")
                break

            # # recover actions using input ----------------------------
            # # udpate pos
            # curr_pos = env.sim.model.site_pos[goal_sid]
            # curr_pos[:] += pos_scale*delta_pos
            # # update rot
            # curr_quat =  env.sim.model.site_quat[goal_sid]
            # curr_quat[:] = mulQuat(euler2quat(rot_scale*delta_euler), curr_quat)
            # # update gripper
            # if delta_gripper !=0:
            #     gripper_state = gripper_scale*delta_gripper # TODO: Update to be delta

            curr_pos = env.sim.model.site_pos[goal_sid]
            curr_pos[:] = trajectory[i_step][:3]
            # update rot
            curr_quat =  env.sim.model.site_quat[goal_sid]
            curr_quat[:] = trajectory[i_step][3:7]
            # update gripper
            gripper_state = trajectory[i_step][7]

            # get action using IK
            ik_result = qpos_from_site_pose(
                        physics = env.sim,
                        site_name = teleop_site,
                        target_pos= curr_pos,
                        target_quat= curr_quat,
                        inplace=False,
                        regularization_strength=1.0)
            if ik_result.success==False:
                print(f"IK(t:{i_step}):: Status:{ik_result.success}, total steps:{ik_result.steps}, err_norm:{ik_result.err_norm}")
            else:
                act[:7] = ik_result.qpos[:7]
                act[7:] = gripper_state
                if action_noise:
                    act = act + env.env.np_random.uniform(high=action_noise, low=-action_noise, size=len(act)).astype(act.dtype)
                if env.normalize_act:
                    act = env.env.robot.normalize_actions(act)

            # nan actions for last log entry
            act = np.nan*np.ones(env.action_space.shape) if i_step == horizon else act

            # log values at time=t ----------------------------------
            datum_dict = dict(
                    time=env.time,
                    observations=obs,
                    actions=act.copy(),
                    rewards=rwd,
                    env_infos=env_info,
                    done=done,
                )
            trace.append_datums(group_key=group_key,dataset_key_val=datum_dict)
            # print(f't={env.time:2.2}, a={act}, o={obs[:3]}')

            # step env using action from t=>t+1 ----------------------
            if i_step < horizon: #incase last actions (nans) can cause issues in step
                obs, rwd, done, env_info = env.step(act)

        print("rollout {} end".format(i_rollout))

    # save and close
    env.close()
    trace.save(output, verify_length=True)

    # render video outputs
    if len(camera)>0:
        if camera[0]!="default":
            trace.render(output_dir=".", output_format="mp4", groups=":", datasets=camera, input_fps=1/env.dt)
        elif output_format=="RoboHive":
            trace.render(output_dir=".", output_format="mp4", groups=":", datasets=["env_infos/obs_dict/rgb:left_cam:240x424:2d","env_infos/obs_dict/rgb:right_cam:240x424:2d","env_infos/obs_dict/rgb:top_cam:240x424:2d","env_infos/obs_dict/rgb:Franka_wrist_cam:240x424:2d"], input_fps=1/env.dt)
        elif output_format=="RoboSet":
            trace.render(output_dir=".", output_format="mp4", groups=":", datasets=["data/rgb_left","data/rgb_right","data/rgb_top","data/rgb_wrist"], input_fps=1/env.dt)


if __name__ == '__main__':
    main()
