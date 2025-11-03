# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import numpy as np
import imageio.v3 as iio
from scipy.spatial.transform import Rotation

import torch

from src.config import cfg, update_argparser, update_config

from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.cameras import MiniCam

import viser
import viser.transforms as tf


def matrix2wxyz(R):
    return Rotation.from_matrix(R).as_quat()[[3,0,1,2]]

def wxyz2matrix(wxyz):
    return Rotation.from_quat(wxyz[[1,2,3,0]]).as_matrix()


class SVRasterViewer:
    def __init__(self, cfg):

        # Load cameras
        data_pack = DataPack(cfg.data, camera_params_only=True)
        self.tr_cam_lst = data_pack.get_train_cameras()
        self.te_cam_lst = data_pack.get_test_cameras()

        # Load model
        self.voxel_model = SparseVoxelModel(cfg.model)
        self.voxel_model.load_iteration(args.iteration)
        self.voxel_model.freeze_vox_geo()

        # Create viser server
        self.server = viser.ViserServer(port=cfg.port)
        self.is_connected = False

        self.server.gui.set_panel_label("SVRaster viser")
        self.server.gui.add_markdown('''
        View control:
        - Mouse drag + scroll
        - WASD + QE keys
        ''')
        self.fps = self.server.gui.add_text("Rending FPS", initial_value="-1", disabled=True)

        # Create gui for setup viewer
        self.active_sh_degree_slider = self.server.gui.add_slider(
            "active_sh_degree",
            min=0,
            max=self.voxel_model.max_sh_degree,
            step=1,
            initial_value=self.voxel_model.active_sh_degree,
        )

        self.ss_slider = self.server.gui.add_slider(
            "ss",
            min=0.5,
            max=2.0,
            step=0.05,
            initial_value=self.voxel_model.ss,
        )

        self.width_slider = self.server.gui.add_slider(
            "width",
            min=64,
            max=2048,
            step=8,
            initial_value=1024,
        )

        self.fovx_slider = self.server.gui.add_slider(
            "fovx",
            min=10,
            max=150,
            step=1,
            initial_value=70,
        )

        self.near_slider = self.server.gui.add_slider(
            "near",
            min=0.02,
            max=10,
            step=0.01,
            initial_value=0.02,
        )

        self.render_dropdown = self.server.gui.add_dropdown(
            "render mod",
            options=["all", "rgb only", "depth only", "normal only"],
            initial_value="all",
        )

        self.output_dropdown = self.server.gui.add_dropdown(
            "output",
            options=["rgb", "alpha", "dmean", "dmed", "dmean2n", "dmed2n", "n"],
            initial_value="rgb",
        )

        # Add camera frustrum
        self.tr_frust = []
        self.te_frust = []

        def add_frustum(name, cam, color):
            c2w = cam.c2w.cpu().numpy()
            frame = self.server.scene.add_camera_frustum(
                name,
                fov=cam.fovy,
                aspect=cam.image_width / cam.image_height,
                scale=0.10,
                wxyz=matrix2wxyz(c2w[:3, :3]),
                position=c2w[:3, 3],
                color=color,
                visible=False,
            )
            @frame.on_click
            def _(event: viser.SceneNodePointerEvent):
                print('Select', name)
                target = event.target
                client = event.client
                with client.atomic():
                    client.camera.wxyz = target.wxyz
                    client.camera.position = target.position
            return frame

        for i, cam in enumerate(self.tr_cam_lst):
            self.tr_frust.append(add_frustum(f"/frustum/train/{i:04d}", cam, [0.,1.,0.]))
        for i, cam in enumerate(self.te_cam_lst):
            self.te_frust.append(add_frustum(f"/frustum/test/{i:04d}", cam, [1.,0.,0.]))
        
        self.show_cam_dropdown = self.server.gui.add_dropdown(
            "show cameras",
            options=["none", "train", "test", "all"],
            initial_value="none",
        )
        @self.show_cam_dropdown.on_update
        def _(_):
            for frame in self.tr_frust:
                frame.visible = self.show_cam_dropdown.value in ["train", "all"]
            for frame in self.te_frust:
                frame.visible = self.show_cam_dropdown.value in ["test", "all"]

        # Server listening
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):

            # Init camera
            with client.atomic():
                init_c2w = self.tr_cam_lst[0].c2w.cpu().numpy()
                client.camera.wxyz = matrix2wxyz(init_c2w[:3, :3])
                client.camera.position = init_c2w[:3, 3]

            @client.camera.on_update
            def _(_):
                pass

            # Everyting ready to go
            self.is_connected = True

        # Download current view
        self.download_button = self.server.gui.add_button("Download view")
        @self.download_button.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None

            im, eps = self.render_viser_camera(client.camera)

            client.send_file_download(
                "svraster_viser.png",
                iio.imwrite("<bytes>", im, extension=".png"),
            )

    @torch.no_grad()
    def render_viser_camera(self, camera: viser.CameraHandle):
        width = self.width_slider.value
        height = round(width / camera.aspect)
        fovx_deg = self.fovx_slider.value
        fovy_deg = fovx_deg * height / width
        near = self.near_slider.value

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = wxyz2matrix(camera.wxyz)
        c2w[:3, 3] = camera.position

        minicam = MiniCam(
            c2w,
            fovx=np.deg2rad(fovx_deg),
            fovy=np.deg2rad(fovy_deg),
            width=width,
            height=height,
            near=near,
        )

        self.voxel_model.active_sh_degree = int(self.active_sh_degree_slider.value)

        render_opt = {
            'ss': self.ss_slider.value,
            'output_T': True,
            'output_depth': True,
            'output_normal': True,
        }
        if self.render_dropdown.value == "rgb only":
            render_opt['output_depth'] = False
            render_opt['output_normal'] = False
        elif self.render_dropdown.value == "depth only":
            render_opt['color_mode'] = "dontcare"
            render_opt['output_normal'] = False
        elif self.render_dropdown.value == "normal only":
            render_opt['color_mode'] = "dontcare"
            render_opt['output_depth'] = False

        start = time.time()
        try:
            render_pkg = self.voxel_model.render(minicam, **render_opt)
        except RuntimeError as e:
            print(e)
        torch.cuda.synchronize()
        end = time.time()
        eps = end - start

        if self.output_dropdown.value == "dmean":
            im = viz_tensordepth(render_pkg['depth'][0])
        elif self.output_dropdown.value == "dmed":
            im = viz_tensordepth(render_pkg['depth'][2])
        elif self.output_dropdown.value == "dmean2n":
            depth2normal = minicam.depth2normal(render_pkg['depth'][0])
            im = im_tensor2np(depth2normal * 0.5 + 0.5)
        elif self.output_dropdown.value == "dmed2n":
            depth_med2normal = minicam.depth2normal(render_pkg['depth'][2])
            im = im_tensor2np(depth_med2normal * 0.5 + 0.5)
        elif self.output_dropdown.value == "n":
            im = im_tensor2np(render_pkg['normal'] * 0.5 + 0.5)
        elif self.output_dropdown.value == "alpha":
            im = im_tensor2np(1 - render_pkg["T"].repeat(3, 1, 1))
        else:
            im = im_tensor2np(render_pkg["color"])
        del render_pkg

        return im, eps

    def update(self):
        if not self.is_connected:
            return

        times = []
        for client in self.server.get_clients().values():
            im, eps = self.render_viser_camera(client.camera)
            times.append(eps)
            client.scene.set_background_image(im, format="jpeg")

        if len(times):
            fps = 1 / np.mean(times)
            self.fps.value = f"{round(fps):4d}"



if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Sparse voxels raster visualizer.")
    parser.add_argument('model_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--port", default=7007, type=int)
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Load config
    update_config(os.path.join(args.model_path, 'config.yaml'))

    # Set additional setup for viewer
    cfg.port = args.port

    # Create and run viewer
    svraster_viewer = SVRasterViewer(cfg)

    while True:
        svraster_viewer.update()
        time.sleep(0.003)
