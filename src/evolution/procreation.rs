use std::cmp::{min, max};

use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;

use crate::evolution::crossover::{crossover_layer_biases, crossover_layer_weights};
use crate::evolution::mutation::{mutate_add_layer, mutate_add_connections};
use crate::individual::Individual;
use crate::layer::Layer;
use crate::tensor::Tensor;


pub fn procreate<T: Tensor>(parent_1: &Individual<T>, parent_2: &Individual<T>) -> Individual<T> {
    let num_layers = parent_1.num_layers();
    if num_layers != parent_2.num_layers() {
        panic!("Both parents must have the same number of layers!");
    }
    let mut layers = Vec::with_capacity(num_layers);
    let mut rng = thread_rng();
    let num_cols_p1 = parent_1[0].weights.shape().1;
    let num_cols_p2 = parent_2[0].weights.shape().1;
    let mut num_cols = rng.gen_range(min(num_cols_p1, num_cols_p2)..=max(num_cols_p1, num_cols_p2));
    for (layer_p1, layer_p2) in parent_1.iter().zip(parent_2) {
        let num_rows_p1 = layer_p1.size();
        let num_rows_p2 = layer_p2.size();
        let num_rows = rng.gen_range(min(num_rows_p1, num_rows_p2)..=max(num_rows_p1, num_rows_p2));
        let mut weights = T::zeros((num_rows, num_cols));
        let mut biases = T::zeros((num_rows, 1));
        crossover_layer_weights(&mut weights, &layer_p1.weights, &layer_p2.weights, &mut rng);
        crossover_layer_biases(&mut biases, &layer_p1.biases, &layer_p2.biases, &mut rng);
        let activation = (*[&layer_p1.activation, &layer_p2.activation].choose(&mut rng).unwrap()).clone();
        layers.push(Layer::new(weights, biases, activation));
        num_cols = num_rows;
    }
    mutate_add_layer(&mut layers, &mut rng);
    mutate_add_connections(&mut layers, &mut rng);
    Individual::new(layers, parent_1.get_cost_function().clone())
}


