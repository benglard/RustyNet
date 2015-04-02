use utilities;

pub struct Vol {
    pub sx: usize,
    pub sy: usize,
    pub depth: usize,
    pub w: Vec<f64>,
    pub dw: Vec<f64>
}

impl Vol {

    pub fn new(sx: usize, sy: usize, depth: usize, c: Option<f64>) -> Vol {
        let n = (sx * sy * depth);
        let mut dw: Vec<f64> = utilities::zeros(n);
        let mut w: Vec<f64> = utilities::zeros(n);

        match c {
            Some(c) => {
                w = utilities::constant(n, c as f64);
            }
            None => {
                let scale: f64 = (1.0 / (n as f64)).sqrt();
                w = utilities::random(n, 0.0, scale);
            }
        }

        Vol { sx: sx, sy: sy, depth: depth, w: w, dw: dw }
    }

    pub fn get(&mut self, x: usize, y: usize, d: usize) -> f64 {
        let idx = ((self.sx * y) + x) * self.depth + d;
        self.w[idx]
    }

    pub fn set(&mut self, x: usize, y: usize, d: usize, v: f64) {
        let idx = ((self.sx * y) + x) * self.depth + d;
        self.w[idx] = v;
    }

    pub fn add(&mut self, x: usize, y: usize, d: usize, v: f64) {
        let idx = ((self.sx * y) + x) * self.depth + d;
        self.w[idx] += v;
    }

    pub fn get_grad(&mut self, x: usize, y: usize, d: usize) -> f64 {
        let idx = ((self.sx * y) + x) * self.depth + d;
        self.dw[idx]
    }

    pub fn set_grad(&mut self, x: usize, y: usize, d: usize, v: f64) {
        let idx = ((self.sx * y) + x) * self.depth + d;
        self.dw[idx] = v;
    }

    pub fn add_grad(&mut self, x: usize, y: usize, d: usize, v: f64) {
        let idx = ((self.sx * y) + x) * self.depth + d;
        self.dw[idx] += v;
    }

    pub fn clone_zero(&mut self) -> Vol {
        Vol::new(self.sx, self.sy, self.depth, Some(0.0))
    }

    pub fn add_from(&mut self, v: Vol) {
        let n: i32 = (self.sx * self.sy * self.depth) as i32;
        for idx in (0 .. n) {
            let idxu: usize = idx as usize;
            self.w[idxu] += v.w[idxu];
        }
    }

    pub fn add_scale(&mut self, v: Vol, a: f64) {
        let n: i32 = (self.sx * self.sy * self.depth) as i32;
        for idx in (0 .. n) {
            let idxu: usize = idx as usize;
            self.w[idxu] += a * v.w[idxu];
        }
    }

    pub fn set_const(&mut self, a: f64) {
        for num in (0 .. self.w.len()) {
            self.w[num as usize] = a
        }
    }

}

impl Clone for Vol {

    fn clone(&self) -> Vol {
        let mut v = Vol::new(self.sx, self.sy, self.depth, Some(0.0));
        let n: i32 = (self.sx * self.sy * self.depth) as i32;
        for idx in (0 .. n) {
            let idxu: usize = idx as usize;
            v.w[idxu] = self.w[idxu];
        }
        v
    }

}