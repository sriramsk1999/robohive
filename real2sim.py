# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/robohive
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """
DESC = """
Modified from ee_teleop.py
Real2sim for HOI4D data
"""

from robohive.utils.quat_math import mat2quat, quat2mat, euler2mat
from robohive.utils.inverse_kinematics import qpos_from_site_pose
import numpy as np
import click
import gym
import os

import open3d as o3d
import pickle
import cv2
from glob import glob
from tqdm import tqdm


def load_hoi4d_trajectory(base_path):
    """loads a test trajectory from a hardcoded path.
    The minimum requirement is the hand pose in the
    camera frame (cam2hand) and the camera pose in the world (world2cam).
    """
    cam_trajectory = o3d.io.read_pinhole_camera_trajectory(
        f"{base_path}/3Dseg/output.log"
    )
    K = cam_trajectory.parameters[0].intrinsic.intrinsic_matrix
    cam2camWorld = np.array([i.extrinsic for i in cam_trajectory.parameters])
    camWorld2cam = np.linalg.inv(cam2camWorld)

    cam2hand = []
    # in some frames, the hand might not be visible, filter these out
    idxs = np.array(
        sorted([int(i.split(".")[0]) for i in os.listdir(f"{base_path}/handpose")])
    )
    for idx in idxs:
        f = open(f"{base_path}/handpose/{idx}.pickle", "rb")
        data = pickle.load(f)
        pose = np.eye(4)
        global_rot = data["poseCoeff"][:3]
        global_rot = cv2.Rodrigues(global_rot)[0]
        pose[:3, :3] = global_rot
        pose[:3, 3] = data["trans"]
        cam2hand.append(pose)
        f.close()
    cam2hand = np.array(cam2hand)

    camWorld2hand = camWorld2cam[idxs] @ cam2hand
    return idxs, camWorld2hand


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
    lines = [
        [i, i + 1] for i in range(len(trajectory_points) - 1)
    ]  # Connect points with lines
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
    o3d.visualization.draw_geometries(
        frames + [trajectory_line, start_sphere, end_sphere, origin_frame]
    )


def retarget_hand_trajectory(camWorld2hand, robotWorld2ee):
    """
    Align hand coordinates with end effector.
    retarget the 4x4 extrinsics to quaternion/translation/gripper state
    """
    scale_factor = 0.75
    # TODO: Scale translation more intelligently
    # Scale down the translation of the trajectory
    # so that end effector does not go out of reach of the robot
    camWorld2hand[:, :3, 3] *= scale_factor

    robotWorld2camWorld = robotWorld2ee @ np.linalg.inv(camWorld2hand[0])
    robotWorld2hand = robotWorld2camWorld @ camWorld2hand

    # An arbitrary rotation to align the trajectory correctly with robot
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = euler2mat((0, np.pi, 0))
    robotWorld2hand = robotWorld2hand @ align_rotation

    robot_trajectory_quat = mat2quat(robotWorld2hand[:, :3, :3])
    robot_trajectory_pos = robotWorld2hand[:, :3, 3]

    # TODO: Handle gripper state properly
    robot_gripper_state = np.zeros((robotWorld2hand.shape[0], 1))
    trajectory = np.concatenate(
        [robot_trajectory_pos, robot_trajectory_quat, robot_gripper_state], axis=-1
    )
    return trajectory


def write_real_sim_video(sim_imgs, base_path, valid_idxs, output_path):
    """
    Visualize sim video and real video side-by-side.
    """
    sim_imgs = np.array([cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in sim_imgs])

    real_imgs = np.array(
        [cv2.imread(i) for i in sorted(glob(f"{base_path}/align_rgb/*jpg"))]
    )
    real_imgs = real_imgs[valid_idxs]
    real_imgs = np.array(
        [cv2.resize(i, (sim_imgs.shape[2], sim_imgs.shape[1])) for i in real_imgs]
    )
    real_and_sim = np.array([cv2.hconcat([i, j]) for i, j in zip(real_imgs, sim_imgs)])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, 30, (real_and_sim.shape[2], real_and_sim.shape[1])
    )

    for frame in real_and_sim:
        out.write(frame)
    out.release()


@click.command(help=DESC)
@click.option(
    "-e",
    "--env_name",
    type=str,
    help="environment to load",
    default="rpFrankaRobotiqData-v0",
)
@click.option(
    "-rn",
    "--reset_noise",
    type=float,
    default=0.0,
    help=("Amplitude of noise during reset"),
)
@click.option(
    "-an",
    "--action_noise",
    type=float,
    default=0.0,
    help=("Amplitude of action noise during rollout"),
)
@click.option(
    "-r",
    "--render",
    type=click.Choice(["onscreen", "offscreen", "onscreen+offscreen", "none"]),
    help="visualize onscreen or offscreen",
    default="offscreen",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    help="seed for generating environment instances",
    default=123,
)
@click.option(
    "-gs",
    "--goal_site",
    type=str,
    help="Site that updates as goal using inputs",
    default="ee_target",
)
@click.option(
    "-ts",
    "--teleop_site",
    type=str,
    help="Site used for teleOp/target for IK",
    default="end_effector",
)
@click.option(
    "-ip",
    "--input_path",
    type=str,
    help="Input HOI4D data for real2sim",
    default="/home/sriram.sk/desktop/hoi4d_vid/_data_sriram_hoi4d_hoi4d_data_ZY20210800003_H3_C14_N42_S207_s05_T2_vid/",
)
@click.option(
    "-op",
    "--output_path",
    type=str,
    help="Directory to store real2sim viz",
    default="real2sim_viz",
)
def main(
    env_name,
    reset_noise,
    action_noise,
    render,
    seed,
    goal_site,
    teleop_site,
    input_path,
    output_path,
):
    base_path = input_path
    os.makedirs(output_path, exist_ok=True)

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    env.env.mujoco_render_frames = True if "onscreen" in render else False
    goal_sid = env.sim.model.site_name2id(goal_site)
    env.sim.model.site_rgba[goal_sid][3] = 0.2  # make visible
    # place ee target in a more suitable location / orientation
    env.sim.model.site_pos[goal_sid] = [
        0.4,
        0,
        1.0,
    ]
    env.sim.model.site_quat[goal_sid] = [
        0,
        0,
        0,
        1,
    ]
    env.sim.forward()

    env.reset()
    curr_pos = env.sim.model.site_pos[goal_sid]
    curr_quat = env.sim.model.site_quat[goal_sid]
    ee_pose = np.eye(4)
    ee_pose[:3, :3] = quat2mat(curr_quat)
    ee_pose[:3, 3] = curr_pos
    valid_idxs, camWorld2hand = load_hoi4d_trajectory(base_path)
    trajectory = retarget_hand_trajectory(camWorld2hand, ee_pose)
    horizon = trajectory.shape[0]

    reset_noise = reset_noise * np.random.uniform(
        low=-1, high=1, size=env.init_qpos.shape
    )
    env.reset(reset_qpos=env.init_qpos + reset_noise, blocking=True)

    # recover init state
    obs, rwd, done, env_info = env.forward()
    act = np.zeros(env.action_space.shape)
    gripper_state = 0
    sim_imgs = []

    for i_step in tqdm(range(horizon)):
        curr_pos = env.sim.model.site_pos[goal_sid]
        curr_pos[:] = trajectory[i_step][:3]
        # update rot
        curr_quat = env.sim.model.site_quat[goal_sid]
        curr_quat[:] = trajectory[i_step][3:7]
        # update gripper
        gripper_state = trajectory[i_step][7]

        # get action using IK
        ik_result = qpos_from_site_pose(
            physics=env.sim,
            site_name=teleop_site,
            target_pos=curr_pos,
            target_quat=curr_quat,
            inplace=False,
            regularization_strength=1.0,
        )
        if ik_result.success == False:
            print(
                f"IK(t:{i_step}):: Status:{ik_result.success}, total steps:{ik_result.steps}, err_norm:{ik_result.err_norm}"
            )
        else:
            act[:7] = ik_result.qpos[:7]
            act[7:] = gripper_state
            if action_noise:
                act = act + env.env.np_random.uniform(
                    high=action_noise, low=-action_noise, size=len(act)
                ).astype(act.dtype)
            if env.normalize_act:
                act = env.env.robot.normalize_actions(act)

        # nan actions for last log entry
        act = np.nan * np.ones(env.action_space.shape) if i_step == horizon else act

        # step env using action from t=>t+1 ----------------------
        if i_step < horizon:  # incase last actions (nans) can cause issues in step
            obs, rwd, done, env_info = env.step(act)

        sim_imgs.append(env.get_exteroception()["rgb:left_cam:240x424:2d"])

    sim_imgs = np.array(sim_imgs)
    output_name = base_path.replace("/", "_") + ".mp4"
    write_real_sim_video(
        sim_imgs, base_path, valid_idxs, f"{output_path}/{output_name}"
    )
    # save and close
    env.close()


if __name__ == "__main__":
    main()
