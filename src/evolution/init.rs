use rand::Rng;
use rand::distributions::uniform::SampleRange;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

use crate::activation::{Activation, Registered};
use crate::component::TensorComponent;
use crate::cost_function::CostFunction;
use crate::individual::Individual;
use crate::layer::Layer;
use crate::tensor::Tensor2;


pub fn init_weight<C: TensorComponent>(rng: &mut ThreadRng, can_zero: bool) -> Option<C> {
    if can_zero && *[true, false].choose(rng).unwrap() {
        Some(C::zero())
    } else {
        C::from_f32(rng.gen_range(0f32..1f32))
    }
}


pub fn random_individual<T: Tensor2, R: Clone + SampleRange<usize>>(
    num_layers: usize,
    input_length: usize,
    output_length: usize,
    neurons_per_layer: R,
    rng: &mut ThreadRng,
) -> Individual<T> {
    let mut layers = Vec::with_capacity(num_layers);
    let mut num_neurons_previous = input_length;
    for idx in 0..num_layers {
        let num_neurons = if idx == num_layers - 1 { 
            output_length
        } else {
            rng.gen_range(neurons_per_layer.clone())
        };
        let mut weights = T::zeros((num_neurons, num_neurons_previous));
        for ((_, _), weight) in weights.iter_indexed_mut() {
            *weight = init_weight(rng, true).unwrap();
        }
        let mut biases = T::zeros((num_neurons, 1));
        for ((_, _), bias) in biases.iter_indexed_mut() {
            *bias = init_weight(rng, true).unwrap();
        }
        layers.push(
            Layer::new(weights, biases, Activation::get("relu").unwrap())
        );
        num_neurons_previous = num_neurons;
    }
    Individual::new(layers, CostFunction::default())
}

