from typing import Union, Literal
import numpy as np
import mujoco
import mujoco.viewer

class MujocoRenderUtils:
    def __init__(self, render_fps, sim_dt):
        self.target_velocity = None

        self.vis_smooth_factor = 1.0
        self.ren_smooth_factor = 1.0

        self.vis_cur_vel = np.zeros(3)
        self.ren_cur_vel = np.zeros(3)

        self.mj_data = None
    
    def update(self, target_velocity, mj_data):
        self.target_velocity = target_velocity
        self.mj_data = mj_data
        
    def update_external_rendering(self,
            handle: Union[mujoco.viewer.Handle, mujoco.Renderer],
            ctype: Literal['viewer', 'renderer'],
        ):
        """ Update external rendering handle (viewer or renderer). """

        def add_thick_arrow(geom_elem, pos, vec, rgba, scale=0.7):
            vel_norm = np.linalg.norm(vec)
            display_norm = min(vel_norm * scale, 1.0)

            if display_norm < 0.10:
                mujoco.mjv_initGeom(
                    geom_elem,
                    type=mujoco.mjtGeom.mjGEOM_NONE,
                    size=[0,0,0], pos=pos, mat=np.eye(3).flatten(), rgba=[0,0,0,0]
                )
                return

            mat = np.zeros(9)
            target_quat = np.zeros(4)
            vec_normalized = vec / vel_norm
            mujoco.mju_quatZ2Vec(target_quat, vec_normalized)
            mujoco.mju_quat2Mat(mat, target_quat)
            
            mat = mat.reshape(3, 3)
            mat[:, 2] *= display_norm 
            
            mujoco.mjv_initGeom(
                geom_elem,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[0.02, 0.02, display_norm], # [height, width, length]
                pos=pos,
                mat=mat.flatten(),
                rgba=rgba
            )

        viewer_geom_idx = 0
        if ctype == 'viewer':
            handle.user_scn.ngeom = 0  # reset user scene geometry
        
        if self.target_velocity is not None:
            base_pos_world = self.mj_data.qpos[:3]
            base_quat = self.mj_data.qpos[3:7]
            
            # rendering arrows start position
            offset_body = np.array([0.0, 0.0, 0.2])
            offset_world = np.zeros(3)
            mujoco.mju_rotVecQuat(offset_world, offset_body, base_quat)
            start_pos = base_pos_world + offset_world

            tgt_vel_body = np.array([self.target_velocity[0], self.target_velocity[1], 0.0])
            
            raw_cur_vel_world = self.mj_data.qvel[:3]
            raw_cur_vel = np.zeros(3)
            neg_quat = np.zeros(4)
            mujoco.mju_negQuat(neg_quat, base_quat)
            mujoco.mju_rotVecQuat(raw_cur_vel, raw_cur_vel_world, neg_quat)
            cur_vel_body = np.array([raw_cur_vel[0], raw_cur_vel[1], 0.0])

            # EMA: v_smooth = alpha * v_new + (1 - alpha) * v_old
            # alpha = self.vis_smooth_factor if ctype == 'viewer' else self.ren_smooth_factor
            self.vis_cur_vel = cur_vel_body
            self.ren_cur_vel = cur_vel_body

            tgt_vel_world = np.zeros(3)
            cur_vel_world = np.zeros(3)
            mujoco.mju_rotVecQuat(tgt_vel_world, tgt_vel_body, base_quat)
            if ctype == 'viewer':
                mujoco.mju_rotVecQuat(cur_vel_world, self.vis_cur_vel, base_quat)
            else:
                mujoco.mju_rotVecQuat(cur_vel_world, self.ren_cur_vel, base_quat)

            COLOR_CMD = [0, 1, 0, 1]   # Green 0x00ff00
            COLOR_REAL = [0, 0, 1, 1]  # Blue  0x0000ff

            if ctype == 'viewer':
                # Cmd Arrow
                add_thick_arrow(handle.user_scn.geoms[viewer_geom_idx], start_pos, tgt_vel_world, COLOR_CMD)
                viewer_geom_idx += 1
                # Real Arrow
                add_thick_arrow(handle.user_scn.geoms[viewer_geom_idx], start_pos, cur_vel_world, COLOR_REAL)
                viewer_geom_idx += 1
            else:
                # Renderer Append
                handle.scene.ngeom += 1
                add_thick_arrow(handle.scene.geoms[handle.scene.ngeom - 1], start_pos, tgt_vel_world, COLOR_CMD)
                handle.scene.ngeom += 1
                add_thick_arrow(handle.scene.geoms[handle.scene.ngeom - 1], start_pos, cur_vel_world, COLOR_REAL)

        if ctype == 'viewer':
            handle.user_scn.ngeom = viewer_geom_idx
