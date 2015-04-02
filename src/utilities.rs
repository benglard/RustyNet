extern crate rand;
use rand::Rng;

pub fn zeros(n: usize) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::new();
    for _ in (0 .. n) {
        v.push(0.0);
    }
    v
}

pub fn constant(n: usize, c: f64) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::new();
    for _ in (0 .. n) {
        v.push(c)
    }
    v
}

pub fn random(n: usize, lower: f64, upper: f64) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::new();
    for _ in (0 .. n) {
        v.push(randn(lower, upper))
    }
    v 
}


static mut return_v: bool = false;
static mut v_val: f64 = 0.0;
pub fn guess_random() -> f64 {
    unsafe {
        if return_v {
            return_v = false;
            return v_val;
        }
        let u: f64 = 2.0 * rand::random::<f64>() - 1.0;
        let v: f64 = 2.0 * rand::random::<f64>() - 1.0;
        let r: f64 = u * u + v * v;
        if r == 0.0 || r > 1.0 {
            return guess_random();
        }
        let c: f64 = (-2.0 * r.ln() / r).sqrt();
        v_val = v * c; // cache this
        return_v = true;
        return u * c;
    }
}

pub fn randf(a: i32, b: i32) -> f64 {
    rand::random::<f64>() * (((b - a) + a) as f64)
}
 
pub fn randi(a: i32, b: i32) -> i32 {
    (rand::random::<f64>() * (((b - a) + a) as f64)) as i32
}

pub fn randn(mu: f64, std: f64) -> f64 {
    mu + guess_random() * std
}