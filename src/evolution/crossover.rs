use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

use crate::evolution::init::init_weight;
use crate::tensor::Tensor2;


pub fn crossover_layer_weights<T: Tensor2>(
    new_weights: &mut T,
    weights_p1: &T,
    weights_p2: &T,
    rng: &mut ThreadRng,
) {
    let (rows_p1, cols_p1) = weights_p1.shape().into();
    let (rows_p2, cols_p2) = weights_p2.shape().into();
    for (idx, component) in new_weights.indexed_iter_mut() {
        let row = idx[0];
        let col = idx[1];
        // both parents have a component at row|col
        if row < rows_p1 && col < cols_p1 && row < rows_p2 && col < cols_p2 {
            let weight_p1 = weights_p1[[row, col]];
            let weight_p2 = weights_p2[[row, col]];
            *component = *[weight_p1, weight_p2].choose(rng).unwrap();
        }
        // only parent_1 has a component at row|col
        else if row < rows_p1 && col < cols_p1 {
            *component = weights_p1[[row, col]];
        }
        // only parent_2 has a component at row|col
        else if row < rows_p2 && col < cols_p2 {
            *component = weights_p2[[row, col]];
        }
        // neither has a component at row|col
        else {
            *component = init_weight(rng, true).unwrap()
        }
    }
}


pub fn crossover_layer_biases<T: Tensor2>(
    new_biases: &mut T,
    biases_p1: &T,
    biases_p2: &T,
    rng: &mut ThreadRng,
) {
    let rows_p1 = biases_p1.shape()[0];
    let rows_p2 = biases_p2.shape()[0];
    for (idx, component) in new_biases.indexed_iter_mut() {
        let row = idx[0];
        // both parents have a component at row
        if rows_p1 > row && rows_p2 > row {
            let bias_p1 = biases_p1[[row, 0]];
            let bias_p2 = biases_p2[[row, 0]];
            *component = *[bias_p1, bias_p2].choose(rng).unwrap();
        }
        // only parent_1 has a component at row
        else if rows_p1 > row {
            *component = biases_p1[[row, 0]];
        }
        // only parent_2 has a component at row
        else {
            *component = biases_p2[[row, 0]];
        }
    }
}

