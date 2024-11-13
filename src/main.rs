use number_ai::network::{
    data::{self, print_image},
    Network,
};

fn main() {
    // env::set_var("RUST_BACKTRACE", "1");

    let mut training_data = data::load_data(
        "MNIST_ORG/train-images.idx3-ubyte",
        "MNIST_ORG/train-labels.idx1-ubyte",
        Some(10000),
    );

    // print_image(training_data.first().unwrap());
    let test = training_data.first().unwrap().clone();

    println!("Initializing Network...");
    let mut network = Network::init(vec![784, 30, 10]);

    println!("{:?}", network.feedforward(test.0.clone()));

    println!("Training Network...");
    network.sgd(&mut training_data, 30, 10, 3.0);

    println!("Final output for: ");
    print_image(&test);
    println!("{:?}", network.feedforward(test.0.clone()));
}
