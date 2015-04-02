extern crate RustyNet;

use RustyNet::vol;
use RustyNet::layer::Layer;
use RustyNet::input_layer;
use RustyNet::linear_layer;

fn main() {
    let mut v = vol::Vol::new(28, 28, 1, None);
    println!("{}", v.get(0, 0, 0));
    v.set(0, 0, 0, 5.0);
    println!("{}", v.get(0, 0, 0));
    println!("{}", v.get_grad(0, 0, 0));
    v.set_grad(0, 0, 0, 5.0);
    println!("{}", v.get_grad(0, 0, 0));

    let mut il = input_layer::InputLayer::new(28, 28, 1);
    let out = il.forward(v, true);
    il.backward();

    let mut ll = linear_layer::LinearLayer::new(28, 28, 1, None);
    let out2 = ll.forward(out, true);
    ll.backward();
    println!("{:?}", out2.w[0 as usize]);
}
