// SPDX-License-Identifier: MIT
// Input handling utilities

use std::collections::HashMap;
use cgmath::Vector3;
use cgmath::InnerSpace;
use three_d::*;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum KeyState {
    Pressed,
    Released,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum KeyCode {
    W,
    A,
    S,
    D,
    Up,
    Down,
    Left,
    Right,
    Space,
    C,
    PageUp,
    PageDown,
    F,
    G,
    Home,
}

impl KeyCode {
    pub fn from_event(event: &Event) -> Option<Self> {
        use KeyCode::*;
        match event {
            Event::KeyPress { kind, .. } | Event::KeyRelease { kind, .. } => match kind {
                Key::W => Some(W),
                Key::A => Some(A),
                Key::S => Some(S),
                Key::D => Some(D),
                Key::ArrowUp => Some(Up),
                Key::ArrowDown => Some(Down),
                Key::ArrowLeft => Some(Left),
                Key::ArrowRight => Some(Right),
                Key::Space => Some(Space),
                Key::C => Some(C),
                Key::PageUp => Some(PageUp),
                Key::PageDown => Some(PageDown),
                Key::F => Some(F),
                Key::G => Some(G),
                Key::Home => Some(Home),
                _ => None,
            },
            _ => None,
        }
    }
}

pub struct InputManager {
    key_states: HashMap<KeyCode, KeyState>,
}

impl Default for InputManager {
    fn default() -> Self {
        Self { key_states: HashMap::new() }
    }
}

impl InputManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_key_states(&mut self, events: &[Event]) {
        for event in events {
            if let Some(key_code) = KeyCode::from_event(event) {
                match event {
                    Event::KeyPress { .. } => {
                        self.key_states.insert(key_code, KeyState::Pressed);
                    }
                    Event::KeyRelease { .. } => {
                        self.key_states.insert(key_code, KeyState::Released);
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn is_key_pressed(&self, key_code: KeyCode) -> bool {
        self.key_states
            .get(&key_code)
            .map_or(false, |state| *state == KeyState::Pressed)
    }

    pub fn get_movement_vector(&self) -> Vector3<f32> {
        let mut move_vec = Vector3::new(0.0, 0.0, 0.0);

        if self.is_key_pressed(KeyCode::W) || self.is_key_pressed(KeyCode::Up) {
            move_vec.z += 1.0;
        }
        if self.is_key_pressed(KeyCode::S) || self.is_key_pressed(KeyCode::Down) {
            move_vec.z -= 1.0;
        }

        if self.is_key_pressed(KeyCode::A) {
            move_vec.x -= 1.0;
        }
        if self.is_key_pressed(KeyCode::D) {
            move_vec.x += 1.0;
        }

        if self.is_key_pressed(KeyCode::Space) {
            move_vec.y += 1.0;
        }
        if self.is_key_pressed(KeyCode::C) {
            move_vec.y -= 1.0;
        }

        if move_vec.magnitude2() > 0.0 {
            move_vec = move_vec.normalize();
        }

        move_vec
    }

    pub fn get_tilt_value(&self) -> f32 {
        let mut tilt = 0.0;

        if self.is_key_pressed(KeyCode::Left) {
            tilt -= 1.0;
        }
        if self.is_key_pressed(KeyCode::Right) {
            tilt += 1.0;
        }

        tilt
    }
}

