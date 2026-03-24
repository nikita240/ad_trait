//! Test AD through nalgebra geometric operations
use ad_trait::reverse_ad::adr32::{adr32, GlobalComputationGraph32};
use ad_trait::AD;
use nalgebra as na;
use simba::scalar::{ComplexField, RealField};

fn main() {
    // Test: distance(point, origin) where point = (x, 0)
    // f(x) = sqrt(x^2) = |x|, df/dx = 1 for x > 0
    GlobalComputationGraph32::get().reset();
    let x = adr32::new_variable(3.0, false);
    let y = adr32::constant(0.0);
    let p1 = na::Point2::new(x, y);
    let p2 = na::Point2::new(adr32::constant(0.0), adr32::constant(0.0));
    let d = na::distance(&p1, &p2);
    let grad = d.get_backwards_mode_grad();
    println!("distance((x,0), origin) at x=3: df/dx={} (expect 1.0)", grad.wrt(&x));

    // Test: pivot_to direction angle
    // pivot_to((0,0), (x,1)) -> angle = atan2(1, x)
    // d/dx atan2(1, x) = -1/(x^2+1) = -0.5 at x=1
    GlobalComputationGraph32::get().reset();
    let x = adr32::new_variable(1.0, false);
    let base = na::Point2::new(adr32::constant(0.0), adr32::constant(0.0));
    let target = na::Point2::new(x, adr32::constant(1.0));
    let dir = target.coords - base.coords;
    let unit = na::Unit::new_normalize(dir);
    let uc = na::UnitComplex::rotation_between_axis(&na::Vector2::x_axis(), &unit);
    let angle = uc.angle();
    let grad = angle.get_backwards_mode_grad();
    println!("atan2(1,x) at x=1: d/dx={} (expect -0.5)", grad.wrt(&x));

    // Test: isometry transform_vector
    // iso = rotation by angle a, transform_vector((1,0)) = (cos(a), sin(a))
    // d/da cos(a) = -sin(a), d/da sin(a) = cos(a)
    GlobalComputationGraph32::get().reset();
    let a = adr32::new_variable(core::f32::consts::FRAC_PI_4, false);
    let iso = na::Isometry2::new(na::Vector2::new(adr32::constant(0.0), adr32::constant(0.0)), a);
    let v = na::Vector2::new(adr32::constant(1.0), adr32::constant(0.0));
    let result = iso.transform_vector(&v);
    let grad_x = result.x.get_backwards_mode_grad();
    let grad_y = result.y.get_backwards_mode_grad();
    let expected_dx = -(core::f32::consts::FRAC_PI_4 as f32).sin();
    let expected_dy = (core::f32::consts::FRAC_PI_4 as f32).cos();
    println!("transform_vector cos(a): d/da={} (expect {:.4})", grad_x.wrt(&a), expected_dx);
    println!("transform_vector sin(a): d/da={} (expect {:.4})", grad_y.wrt(&a), expected_dy);

    // Test: full mini cost = distance(transform(input), target)^2
    // This mimics the MPC structure
    GlobalComputationGraph32::get().reset();
    let cmd = adr32::new_variable(0.5, false);
    let measurement: adr32 = adr32::constant(100.0);
    let predicted_len = measurement + cmd * adr32::constant(55.0 * 0.4);
    let target_len: adr32 = adr32::constant(120.0);
    let error = predicted_len - target_len;
    let cost = error * error;
    let grad = cost.get_backwards_mode_grad();
    // cost = (100 + 22*cmd - 120)^2 = (22*cmd - 20)^2
    // d/d_cmd = 2*(22*cmd - 20)*22 = 2*(11 - 20)*22 = 2*(-9)*22 = -396
    let fd = {
        let h = 1e-3f32;
        let c0 = (100.0 + 22.0*0.5 - 120.0).powi(2);
        let c1 = (100.0 + 22.0*(0.5+h) - 120.0).powi(2);
        (c1 - c0) / h
    };
    println!("mini cost: AD={}, FD={:.1} (expect -396)", grad.wrt(&cmd), fd);
}
