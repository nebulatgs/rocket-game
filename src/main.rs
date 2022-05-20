use std::collections::BTreeSet;

use nannou::{geom::Padding, glam::Vec3Swizzles, prelude::*, wgpu::BlendComponent};

#[derive(Debug, Default, Clone)]
struct Camera {
    pub pos: Vec3,
    pub rot: Quat,
    pub scl: Vec3,

    pub vel: Vec3,
    pub acc: Vec3,
    pub drag: f32,
}

#[derive(Debug, Default, Clone)]
struct Planet {
    pub radius: f32,
    pub mass: f32,

    pub pos: Vec3,
    pub vel: Vec3,
    pub acc: Vec3,
}

#[derive(Debug, Default, Clone)]
struct Rocket {
    pub mass: f32,
    pub thrust: f32,
    pub drag: f32,

    pub pos: Vec3,
    pub rot: Quat,

    pub vel: Vec3,
    pub acc: Vec3,

    pub shape: Vec2,
}

#[derive(Debug, Default, Clone)]
struct Model {
    pub camera: Camera,
    pub rocket: Rocket,
    pub planets: Vec<Planet>,
    pub keys: BTreeSet<Key>,
    pub timescale: f32,
}

impl Model {
    const G: f64 = 6.67408e-11;
}

fn main() {
    nannou::app(model)
        .event(event)
        .update(update)
        .simple_window(view)
        .run();
}

fn model(_app: &App) -> Model {
    let main_planet = Planet {
        radius: 1.0e3,
        mass: 5.0e4,
        vel: vec3(0.0, -2.0e1, 0.0),
        ..Default::default()
    };

    let moon = Planet {
        radius: 4.0e2,
        mass: 5.0e3,
        pos: vec3(1.0e5, 0.0, 0.0),
        vel: vec3(0.0, 2.0e2, 0.0),
        ..Default::default()
    };

    let mut rocket = Rocket {
        mass: 10.0,
        thrust: 200.0,
        drag: 0.0,

        pos: vec3(0.0, main_planet.radius, 0.0),
        rot: Quat::IDENTITY,

        vel: vec3(0.0, 0.0, 0.0),
        acc: vec3(0.0, 0.0, 0.0),

        shape: vec2(20.0, 40.0),
    };

    rocket.pos.y += rocket.shape.y / 2.0;

    let camera = Camera {
        // pos: vec3(0.0, -main_planet.radius, 1.0),
        pos: vec3(-1.0e5, 0.0, 0.0),
        rot: Quat::IDENTITY,
        scl: vec3(0.01, 0.01, 1.0),

        vel: vec3(0.0, 0.0, 0.0),
        acc: vec3(0.0, 0.0, 0.0),
        drag: 10.0,
    };

    let planets = vec![main_planet, moon];

    Model {
        camera,
        planets,
        rocket,
        timescale: 10.0,
        ..Default::default()
    }
}

fn event(_app: &App, model: &mut Model, event: Event) {
    match event {
        Event::WindowEvent {
            simple: Some(win_event),
            ..
        } => match win_event {
            WindowEvent::KeyPressed(key) => {
                match key {
                    Key::Right => model.timescale += 10.0,
                    Key::Left => model.timescale = (model.timescale - 10.0).clamp(5.0, 100.0),
                    _ => {}
                }
                model.keys.insert(key);
            }
            WindowEvent::KeyReleased(ref key) => {
                model.keys.remove(key);
            }
            WindowEvent::MouseWheel(MouseScrollDelta::LineDelta(x, y), _) => {
                model.camera.scl *= 1.0 - y as f32 * 0.1;
            }
            _ => {}
        },
        _ => {}
    }
}

fn handle_keys(app: &App, model: &mut Model, dt: f32) {
    if model.keys.contains(&Key::Space) {
        let thrust = vec3(0.0, model.rocket.thrust, 0.0);
        let r_thrust = model.rocket.rot.mul_vec3(thrust);
        let acc = r_thrust / model.rocket.mass;
        model.rocket.acc += acc * dt;
    }
    if model.keys.contains(&Key::D) {
        let rot = Quat::from_axis_angle(vec3(0.0, 0.0, 1.0), 1.5 * -PI / 180.0);
        model.rocket.rot = model.rocket.rot * rot;
    }
    if model.keys.contains(&Key::A) {
        let rot = Quat::from_axis_angle(vec3(0.0, 0.0, 1.0), 1.5 * PI / 180.0);
        model.rocket.rot = model.rocket.rot * rot;
    }
}

fn simulate(model: &mut Model, dt: f32) {
    for planet_ind in 0..model.planets.len() {
        for other_planet_ind in 0..model.planets.len() {
            if planet_ind == other_planet_ind {
                continue;
            }
            let planet_force = {
                let planet = &model.planets[planet_ind];
                let other_planet = &model.planets[other_planet_ind];

                let dist = other_planet.pos.distance(planet.pos);
                let f = (other_planet.mass * planet.mass) / dist;
                let f_vec = (planet.pos - other_planet.pos).normalize() * f;
                f_vec / planet.mass * -1.0
            };
            {
                let planet = &mut model.planets[planet_ind];
                planet.acc += planet_force;
            }
        }
        {
            let planet = &mut model.planets[planet_ind];

            let dist = model.rocket.pos.distance(planet.pos);
            let f = (model.rocket.mass * planet.mass) / dist;
            let f_vec = (planet.pos - model.rocket.pos).normalize() * f;

            model.rocket.acc += f_vec / model.rocket.mass;
            // planet.acc -= f_vec / planet.mass;

            planet.vel += planet.acc * dt;
            planet.pos += planet.vel * dt;

            planet.acc *= 0.0;
        }
    }
}

fn update_camera(app: &App, model: &mut Model, dt: f32) {
    model.camera.vel = model.planets[1].vel;

    let dims = app.window_rect().xy_wh();
    let rect = Rect::from_xy_wh(dims.0, dims.1 / model.camera.scl.xy())
        .relative_to([model.camera.pos.x, model.camera.pos.y]);
    if !rect
        .pad(5.0)
        .pad_top(model.rocket.shape.y)
        .pad_bottom(model.rocket.shape.y)
        .pad_left(model.rocket.shape.x)
        .pad_right(model.rocket.shape.x)
        .contains_point([model.rocket.pos.x, model.rocket.pos.y])
    {
        let top_right = rect.top_right();

        let cam_pos = model.camera.pos.xy() * vec2(-1.0, -1.0);

        let cam_corner_vec = top_right - cam_pos;
        let rocket_cam_vec = (model.rocket.pos.xy() - cam_pos).abs();
        let rocket_corner_vec = (cam_corner_vec - rocket_cam_vec).abs();

        let bound_dist = cam_corner_vec - (rocket_cam_vec + rocket_corner_vec);
        let bounded = cam_corner_vec - (bound_dist / 2.0);
        let fac = (cam_corner_vec / bounded).min_element();
        // dbg!(bound_dist, fac, model.camera.scl);
        // let ref_len = cam_pos.distance(top_right);

        // let len = model.rocket.pos.abs().xy().distance(cam_pos) / ref_len;
        // dbg!(
        //     cam_pos,
        //     top_right,
        //     ref_len,
        //     model.rocket.pos.abs().xy().distance(cam_pos),
        //     len
        // );
        model.camera.scl *= vec3(fac, fac, 1.0);
    }

    model.camera.pos -= model.camera.vel * dt;
    // dbg!(model.camera.pos);
    // model.camera.vel /= 1.0 + model.camera.drag * dt;
    // // model.camera.vel += model.camera.acc * dt;
    // // model.camera.acc = vec3(0.0, 0.0, 0.0);
}

fn update(app: &App, model: &mut Model, update: Update) {
    let dt = update.since_last.as_secs_f32() * model.timescale;

    handle_keys(app, model, dt);
    simulate(model, dt);

    // Update rocket
    update_rocket(model, dt);

    update_camera(app, model, dt);
}

fn update_rocket(model: &mut Model, dt: f32) {
    let rocket = &mut model.rocket;
    rocket.vel += rocket.acc * dt;
    rocket.pos += rocket.vel * dt;
    rocket.acc = vec3(0.0, 0.0, 0.0);
    rocket.vel /= 1.0 + rocket.drag * dt;
}

fn generate_predicted_paths(model: &Model) -> Vec<Vec<(Vec2, Hsla)>> {
    let step_size = 1.0;
    let iterations = (10.0 / model.camera.scl.x).clamp(5.0, 100.0) as usize;

    let mut p_model = model.clone();
    let mut planet_paths: Vec<Vec<(Vec2, Hsla)>> = vec![vec![]; p_model.planets.len()];
    let mut rocket_path = Vec::new();

    for i in (0..iterations).rev() {
        simulate(&mut p_model, step_size);
        update_rocket(&mut p_model, step_size);
        rocket_path.push((
            p_model.rocket.pos.xy(),
            hsla(
                (p_model.rocket.vel.length() / 1000.0).clamp(0.0, 0.5),
                1.0,
                0.5,
                i as f32 / iterations as f32,
            ),
        ));
        for (j, planet) in p_model.planets.iter().enumerate() {
            planet_paths[j].push((
                planet.pos.xy(),
                hsla(
                    (planet.vel.length() / 1000.0).clamp(0.0, 0.5),
                    1.0,
                    0.5,
                    i as f32 / iterations as f32,
                ),
            ));
        }
    }

    planet_paths.push(rocket_path);
    planet_paths
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app
        .draw()
        .scale_axes(model.camera.scl)
        .translate(model.camera.pos)
        .alpha_blend(BlendComponent::OVER);

    draw.background().color(hsl(0.72, 0.12, 0.01));
    for planet in &model.planets {
        draw.ellipse()
            .radius(planet.radius)
            .xyz(planet.pos)
            .color(RED);
    }

    let scaled = model.rocket.shape / model.camera.scl.xy();
    draw.polygon()
        .points([
            (-(scaled.x / 2.0), -(scaled.y / 2.0)),
            (0.0, scaled.y / 2.0),
            (scaled.x / 2.0, -(scaled.y / 2.0)),
        ])
        // .wh(model.rocket.shape / model.camera.scl.xy())
        .xyz(model.rocket.pos)
        .quaternion(model.rocket.rot)
        .color(BLUE);
    for path in generate_predicted_paths(model) {
        draw.polyline()
            .stroke_weight(2.0 / model.camera.scl.x)
            .caps_round()
            .points_colored(path);
    }
    draw.to_frame(app, &frame).unwrap();
}
