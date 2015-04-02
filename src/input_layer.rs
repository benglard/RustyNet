use layer;
use vol;
use utilities;
use std::string;

pub struct InputLayer {
    pub out_sx: usize,
    pub out_sy: usize,
    pub out_depth: usize,
    pub layer_type: string::String,
    pub in_act: vol::Vol,
    pub out_act: vol::Vol
}

impl InputLayer {

    pub fn new(in_sx: usize, in_sy: usize, in_depth: usize) -> InputLayer {
        InputLayer {
            out_sx: in_sx,
            out_sy: in_sy,
            out_depth: in_depth,
            layer_type: string::String::from_str("input"),
            in_act: vol::Vol::new(in_sx, in_sy, in_depth, Some(0.0)),
            out_act: vol::Vol::new(in_sx, in_sy, in_depth, Some(0.0))
        }
    }

}

impl layer::Layer for InputLayer {

    fn forward(&mut self, v: vol::Vol, is_training: bool) -> vol::Vol {
        self.in_act = v.clone();
        self.out_act = v.clone();
        self.out_act.clone()
    }

    fn backward(&mut self) {}

}