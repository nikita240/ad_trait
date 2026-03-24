use ad_trait::reverse_ad::adr32::{adr32, GlobalComputationGraph32};
use ad_trait::AD;

fn main() {
    // Test 1: f(x,y) = x*y, gradient should be (y, x) = (3, 2)
    GlobalComputationGraph32::get().reset();
    let x = adr32::new_variable(2.0, false);
    let y = adr32::new_variable(3.0, false);
    let f = x * y;
    let grad = f.get_backwards_mode_grad();
    println!("Test x*y at (2,3): df/dx={} (expect 3), df/dy={} (expect 2)", 
             grad.wrt(&x), grad.wrt(&y));

    // Test 2: f(x) = x^2, gradient should be 2x = 6
    GlobalComputationGraph32::get().reset();
    let x = adr32::new_variable(3.0, false);
    let f = x * x;
    let grad = f.get_backwards_mode_grad();
    println!("Test x^2 at x=3: df/dx={} (expect 6)", grad.wrt(&x));

    // Test 3: f(x) = sin(x), gradient should be cos(x) = cos(1) ≈ 0.5403
    GlobalComputationGraph32::get().reset();
    let x = adr32::new_variable(1.0, false);
    let f = simba::scalar::ComplexField::sin(x);
    let grad = f.get_backwards_mode_grad();
    println!("Test sin(x) at x=1: df/dx={} (expect ~0.5403)", grad.wrt(&x));

    // Test 4: f(x,y) = sqrt(x*x + y*y), gradient should be (x/r, y/r)
    GlobalComputationGraph32::get().reset();
    let x = adr32::new_variable(3.0, false);
    let y = adr32::new_variable(4.0, false);
    let f = simba::scalar::ComplexField::sqrt(x*x + y*y);
    let grad = f.get_backwards_mode_grad();
    println!("Test sqrt(x²+y²) at (3,4): df/dx={} (expect 0.6), df/dy={} (expect 0.8)", 
             grad.wrt(&x), grad.wrt(&y));

    // Test 5: f(x) = x + constant, gradient should be 1
    GlobalComputationGraph32::get().reset();
    let x = adr32::new_variable(5.0, false);
    let c = adr32::constant(10.0);
    let f = x + c;
    let grad = f.get_backwards_mode_grad();
    println!("Test x+const at x=5: df/dx={} (expect 1)", grad.wrt(&x));

    // Test 6: chain - f(x) = (x+1)*(x+1), gradient should be 2*(x+1) = 8
    GlobalComputationGraph32::get().reset();
    let x = adr32::new_variable(3.0, false);
    let c = adr32::constant(1.0);
    let t = x + c;
    let f = t * t;
    let grad = f.get_backwards_mode_grad();
    println!("Test (x+1)^2 at x=3: df/dx={} (expect 8)", grad.wrt(&x));

    // Test 7: atan2
    GlobalComputationGraph32::get().reset();
    let y = adr32::new_variable(1.0, false);
    let x = adr32::new_variable(1.0, false);
    let f = simba::scalar::RealField::atan2(y, x);
    let grad = f.get_backwards_mode_grad();
    // d/dy atan2(y,x) = x/(x²+y²) = 1/2 = 0.5
    // d/dx atan2(y,x) = -y/(x²+y²) = -1/2 = -0.5
    println!("Test atan2(y,x) at (1,1): df/dy={} (expect 0.5), df/dx={} (expect -0.5)", 
             grad.wrt(&y), grad.wrt(&x));
}
