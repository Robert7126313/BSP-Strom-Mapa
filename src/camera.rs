// SPDX-License-Identifier: MIT
// Camera utilities

use crate::config::CONFIG;
use crate::input::{InputManager, KeyCode};
use cgmath::{Deg, Vector3};
use std::f32::consts::FRAC_PI_2;
use three_d::*;

#[derive(Clone)]
pub struct FreeCamera {
    pub pos: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
    pub look_speed: f32,
}

impl FreeCamera {
    pub fn new(pos: Vector3<f32>) -> Self {
        let cfg = CONFIG.lock().unwrap().clone();
        Self {
            pos,
            yaw: -FRAC_PI_2,
            pitch: 0.0,
            speed: cfg.camera_speed,
            look_speed: cfg.look_speed,
        }
    }

    pub fn dir(&self) -> Vector3<f32> {
        Vector3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    pub fn right(&self) -> Vector3<f32> {
        self.dir().cross(Vector3::unit_y()).normalize()
    }

    pub fn update_smooth(&mut self, input_manager: &InputManager, dt: f32) {
        let raw_move_vec = input_manager.get_movement_vector();
        let tilt_value = input_manager.get_tilt_value();

        let mut v = Vector3::new(0.0, 0.0, 0.0);
        if raw_move_vec.z != 0.0 {
            v += self.dir() * raw_move_vec.z;
        }
        if raw_move_vec.x != 0.0 {
            v += self.right() * raw_move_vec.x;
        }
        if raw_move_vec.y != 0.0 {
            v += Vector3::unit_y() * raw_move_vec.y;
        }
        if v.magnitude2() > 0.0 {
            self.pos += v * self.speed * dt;
        }
        if tilt_value != 0.0 {
            self.yaw += tilt_value * self.look_speed * dt;
        }
        if input_manager.is_key_pressed(KeyCode::Up) {
            self.pitch += self.look_speed * dt;
        }
        if input_manager.is_key_pressed(KeyCode::Down) {
            self.pitch -= self.look_speed * dt;
        }
        // Wrap pitch to keep it within [-PI, PI] allowing full 360° rotation
        self.pitch = (self.pitch + std::f32::consts::PI) % (2.0 * std::f32::consts::PI)
            - std::f32::consts::PI;
    }

    pub fn cam(&self, vp: Viewport) -> Camera {
        let cfg = CONFIG.lock().unwrap().clone();
        Camera::new_perspective(
            vp,
            self.pos,
            self.pos + self.dir(),
            Vector3::unit_y(),
            Deg(cfg.default_fov_deg),
            cfg.near_plane,
            cfg.far_plane,
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CamMode {
    Spectator,
    ThirdPerson,
}

#[derive(Clone)]
pub struct CameraState {
    pub pos: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
}

impl CameraState {
    pub fn new(pos: Vector3<f32>) -> Self {
        let speed = CONFIG.lock().unwrap().camera_speed;
        Self {
            pos,
            yaw: -FRAC_PI_2,
            pitch: 0.0,
            speed,
        }
    }

    pub fn from_camera(camera: &FreeCamera) -> Self {
        Self {
            pos: camera.pos,
            yaw: camera.yaw,
            pitch: camera.pitch,
            speed: camera.speed,
        }
    }

    pub fn apply_to_camera(&self, camera: &mut FreeCamera) {
        camera.pos = self.pos;
        camera.yaw = self.yaw;
        camera.pitch = self.pitch;
        camera.speed = self.speed;
    }
}
