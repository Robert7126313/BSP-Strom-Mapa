// SPDX-License-Identifier: MIT
// Camera utilities

use cgmath::{Vector3, Deg};
use three_d::*;
use crate::input::{InputManager, KeyCode};

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
        Self { pos, yaw: -FRAC_PI_2, pitch: 0.0, speed: 4.0, look_speed: 2.0 }
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
            self.pitch = (self.pitch + self.look_speed * dt).clamp(-1.5, 1.5);
        }
        if input_manager.is_key_pressed(KeyCode::Down) {
            self.pitch = (self.pitch - self.look_speed * dt).clamp(-1.5, 1.5);
        }
    }

    pub fn cam(&self, vp: Viewport) -> Camera {
        Camera::new_perspective(
            vp,
            self.pos,
            self.pos + self.dir(),
            Vector3::unit_y(),
            Deg(60.0),
            0.1,
            1000.0,
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CamMode {
    Spectator,
    ThirdPerson,
}

pub struct SwitchDelay {
    last_switch_time: f64,
    cooldown: f64,
}

impl SwitchDelay {
    pub fn new(cooldown: f64) -> Self {
        Self { last_switch_time: 0.0, cooldown }
    }
    pub fn can_switch(&self, current_time: f64) -> bool {
        current_time - self.last_switch_time >= self.cooldown
    }
    pub fn record_switch(&mut self, current_time: f64) {
        self.last_switch_time = current_time;
    }
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
        Self { pos, yaw: -FRAC_PI_2, pitch: 0.0, speed: 4.0 }
    }

    pub fn from_camera(camera: &FreeCamera) -> Self {
        Self { pos: camera.pos, yaw: camera.yaw, pitch: camera.pitch, speed: camera.speed }
    }

    pub fn apply_to_camera(&self, camera: &mut FreeCamera) {
        camera.pos = self.pos;
        camera.yaw = self.yaw;
        camera.pitch = self.pitch;
        camera.speed = self.speed;
    }
}

pub fn reset_camera_to_default(camera: &mut FreeCamera, default_position: Vector3<f32>, speed: f32) {
    let mut reset_state = CameraState::new(default_position);
    reset_state.speed = speed;
    reset_state.apply_to_camera(camera);
}

