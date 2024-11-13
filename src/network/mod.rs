use faer::Mat;
use itertools::Itertools;
use rand::{self, seq::SliceRandom, thread_rng};

mod mat_func;
use mat_func::{cost_prime, hadamard, sig_prime_mat, MatFunc, VecMatFunc};

pub mod data;
use data::{label_to_target, Data, Label};

pub struct Network {
    nlayer: usize,
    layer_sizes: Vec<usize>,
    weights: Vec<Mat<f32>>,
    bias: Vec<Mat<f32>>,
}

impl Network {
    pub fn init(layer_sizes: Vec<usize>) -> Self {
        if layer_sizes.len() < 2 {
            panic!(
                "Insufficient layers, need at least 2, currently has {} layers\n",
                layer_sizes.len()
            );
        }

        Network {
            nlayer: layer_sizes.len(),
            weights: gen_weights(&layer_sizes),
            bias: gen_bias(&layer_sizes),
            layer_sizes,
        }
    }

    // takes in input and outputs the result
    pub fn feedforward(&self, mut input: Mat<f32>) -> Mat<f32> {
        assert_eq!(
            input.nrows(),
            self.layer_sizes[0],
            "Starting input nrow does not match network's"
        );
        assert_eq!(
            input.ncols(),
            1,
            "Starting input should be an N by 1 matrix"
        );

        // weights and bias has one less index since no values for first layer
        // for layer in 0..=(self.nlayer - 2) {
        //     input = (&self.weights[layer] * &input) + &self.bias[layer];
        //     input.sigmoid_mat();
        // }
        for (weight, bias) in self.weights.iter().zip_eq(self.bias.iter()) {
            input = (weight * input) + bias;
            input.sigmoid_mat();
        }

        input
    }

    // feed forward and outputs the a and z values in each layer
    // excludes first layer
    fn feedforward_az(&self, mut input: Mat<f32>) -> (Vec<Mat<f32>>, Vec<Mat<f32>>) {
        assert_eq!(
            input.nrows(),
            self.layer_sizes[0],
            "Starting input nrow {} does not match network's",
            input.nrows()
        );
        assert_eq!(
            input.ncols(),
            1,
            "Starting input ncol {} should be one.",
            input.ncols()
        );

        let mut a = vec![];
        let mut z = vec![];

        for (weight, bias) in self.weights.iter().zip_eq(self.bias.iter()) {
            input = (weight * input) + bias;
            z.push(input.clone());

            input.sigmoid_mat();
            a.push(input.clone());
        }

        (a, z)
    }

    pub fn sgd(
        &mut self,
        training_data: &mut Vec<(Data, Label)>,
        epochs: u32,
        batch_size: u32,
        eta: f32,
    ) {
        for epoch in 0..epochs {
            training_data.shuffle(&mut thread_rng());

            for (number, batch) in training_data.chunks(batch_size as usize).enumerate() {
                println!("Batch number {}", number + 1);
                self.update_batch(batch, eta);
            }

            println!("Epoch {epoch} completed.");
        }
    }

    fn update_batch(&mut self, batch: &[(Data, Label)], eta: f32) {
        let mut scaled_dw: Vec<Mat<f32>> = self
            .weights
            .iter()
            .map(|w| Mat::<f32>::zeros(w.nrows(), w.ncols()))
            .collect();

        let mut scaled_db: Vec<Mat<f32>> = self
            .weights
            .iter()
            .map(|w| Mat::<f32>::zeros(w.nrows(), 1))
            .collect();

        // sums up all delta weights and biases for each data point in the given batch
        for (input, target) in batch {
            let (dw, db) = self.backprop(input, target);

            scaled_dw.add_vmat(&dw);
            scaled_db.add_vmat(&db);
        }

        // take average of delta weights and biases by dividing by batch size
        // multiply result by eta to get scaled deltas
        scaled_dw.mult_v_mat_by_n(-eta / batch.len() as f32);
        scaled_db.mult_v_mat_by_n(-eta / batch.len() as f32);

        self.weights.add_vmat(&scaled_dw);
        self.bias.add_vmat(&scaled_db);
    }

    // returns delta weights and bias for each layer excluding the first
    fn backprop(&self, input: &Data, target: &Label) -> (Vec<Mat<f32>>, Vec<Mat<f32>>) {
        // index for layers starting from 0 => layer 2
        // weights and bias has one less length
        let l = self.nlayer - 2;

        // instantiate the delta vectors with empty matricies of size l+1 so that the vector can be indexed
        let mut dw_vec = Vec::from_iter(std::iter::repeat_n(Mat::new(), l + 1));
        let mut db_vec = Vec::from_iter(std::iter::repeat_n(Mat::new(), l + 1));

        // clones input as feedforward_z works directly with mutable input
        // note both a and z are vectors
        let (a, z) = self.feedforward_az(input.clone());

        // takes output error of final layer
        let mut delta = hadamard(
            &cost_prime(&a[l], &label_to_target(*target)),
            &sig_prime_mat(&z[l]),
        );
        db_vec[l] = delta.clone();
        dw_vec[l] = delta.clone() * a[l - 1].transpose();

        // start backpropagation
        // moves from l-1 to l-2 ... to l-l = 0
        for layer in (1..=(l - 1)).rev() {
            // delta here is previous delta (l+1)
            delta = hadamard(
                &(self.weights[layer + 1].transpose() * delta),
                &sig_prime_mat(&z[layer]),
            );
            db_vec[layer] = delta.clone();
            dw_vec[layer] = delta.clone() * a[layer - 1].transpose();
        }

        // for last layer 0
        delta = hadamard(
            &(self.weights[1].transpose() * delta),
            &sig_prime_mat(&z[0]),
        );
        db_vec[0] = delta.clone();
        // for last layer, a is the same as input
        dw_vec[0] = delta.clone() * input.transpose();

        (dw_vec, db_vec)
    }
}

fn gen_bias(layer_sizes: &[usize]) -> Vec<Mat<f32>> {
    let mut output = vec![];

    // no weights/bias given to input layer
    for size in layer_sizes.iter().skip(1) {
        output.push(Mat::<f32>::from_fn(*size, 1, |_, _| rand::random::<f32>()));
    }

    output
}

fn gen_weights(layer_sizes: &[usize]) -> Vec<Mat<f32>> {
    let mut output = vec![];

    // no weights/bias given to input layer
    let mut prev_size = layer_sizes[0];
    for size in layer_sizes.iter().skip(1) {
        output.push(Mat::<f32>::from_fn(*size, prev_size, |_, _| {
            rand::random::<f32>()
        }));
        prev_size = *size;
    }

    output
}
