use layer;
use vol;
use utilities;
use std::string;

pub struct LinearLayer {
    pub num_inputs: usize,
    pub out_sx: usize,
    pub out_sy: usize,
    pub out_depth: usize,
    pub bias: f64,
    pub layer_type: string::String,
    pub in_act: vol::Vol,
    pub out_act: vol::Vol,
    pub filters: Vec<vol::Vol>,
    pub biases: vol::Vol
}

impl LinearLayer {

    pub fn new(in_sx: usize, in_sy: usize, in_depth: usize, bias_pref: Option<f64>) -> LinearLayer {
        let num_inputs = in_sx * in_sy * in_depth;
        let mut filters: Vec<vol::Vol> = Vec::new();
        for _ in (0 .. num_inputs) {
            filters.push(vol::Vol::new(1, 1, num_inputs, None));
        }

        let bias = match bias_pref {
            Some(bias_pref) => { bias_pref }
            None => { 0.0 }
        };

        let mut biases = vol::Vol::new(1, 1, num_inputs, Some(bias));

        LinearLayer {
            num_inputs: num_inputs,
            out_sx: 1,
            out_sy: 1,
            out_depth: in_sx * in_sy * in_depth,
            bias: 0.0,
            layer_type: string::String::from_str("linear"),
            in_act: vol::Vol::new(in_sx, in_sy, in_depth, Some(0.0)),
            out_act: vol::Vol::new(in_sx, in_sy, in_depth, Some(0.0)),
            filters: filters,
            biases: biases
        }
    }

}

impl layer::Layer for LinearLayer {

    fn forward(&mut self, v: vol::Vol, is_training: bool) -> vol::Vol {
        self.in_act = v.clone();
        let mut A = vol::Vol::new(1, 1, self.out_depth, Some(0.0));
        let v_w = v.w;

        // dot(W, x) + b
        for i in (0 .. self.out_depth) {
            let iu = i as usize;
            let mut sum_a = 0.0;
            let fiw = self.filters[iu].w.clone();
            for d in (0 .. self.num_inputs) {
                let du = d as usize;
                sum_a += v_w[du] * fiw[du]; 
            }
            sum_a += self.biases.w[iu];
            A.w[iu] = sum_a;
        }

        self.out_act = A.clone();
        A.clone()
    }

    fn backward(&mut self) {
        let mut V = self.in_act.clone();
        V.dw = utilities::zeros(V.w.len());

        // compute gradient wrt weights and data
        for i in (0 .. self.out_depth) {
            let iu = i as usize;
            let mut fi = self.filters[iu].clone();
            let chain_grad = self.out_act.dw.clone()[iu];

            for d in (0 .. self.num_inputs) {
                let du = d as usize;
                V.dw[du] += fi.w[du] * chain_grad; //grad wrt input data
                fi.dw[du] += V.w[du] * chain_grad; //grad wrt params
            }

            self.biases.dw.clone()[iu] += chain_grad
        }
    }

}