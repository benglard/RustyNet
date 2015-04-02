use vol;

pub trait Layer {
    // All layers implement a forward() and backward() fn's
    fn forward(&mut self, v: vol::Vol, is_training: bool) -> vol::Vol;
    fn backward(&mut self);
}