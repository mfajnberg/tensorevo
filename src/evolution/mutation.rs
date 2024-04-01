use num_traits::{One, Zero};
use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::rngs::ThreadRng;
use rand_distr::{Distribution, Poisson};

use crate::evolution::init::init_weight;
use crate::tensor::Tensor2;
use crate::layer::Layer;


pub fn mutate_add_layer<T: Tensor2>(layers: &mut Vec<Layer<T>>, rng: &mut ThreadRng) {
    if !should_add_layer(layers.len(), rng) { return; }

    let new_layer_idx = rng.gen_range(1..=layers.len());
    let following_layer = &mut layers[new_layer_idx];
    // columns in new layer's weight matrix = neurons in previous layers
    // = columns in weight matrix of following layer BEFORE mutation
    let (following_weights_num_rows, following_weights_num_cols) = following_layer.weights.shape();
    // Next layer weight matrix BEFORE mutation:
    // 1  0
    // 2  3
    // 0  4
    // New layer weight matrix:
    // 1  0    (1)
    // 1  0    (2)
    // 0  1    (3)
    // 0  1    (4)
    // Next layer weight matrix AFTER mutation:
    // 1  0  0  0
    // 0  2  3  0
    // 0  0  0  4

    // rows in new layer's weight matrix = neurons = connections between
    // = non-zero entries in weight matrix of following layer BEFORE mutation
    let zero_component = T::Component::zero();
    let new_weights_num_rows = following_layer.weights.iter().filter(|&weight| *weight != zero_component).count();
    let mut new_weights = T::zeros([new_weights_num_rows, following_weights_num_cols]);
    // modified following weight matrix:
    // rows unchanged, columns = rows in new layer's weight matrix
    let mut following_weights = T::zeros([following_weights_num_rows, new_weights_num_rows]);
    let mut connection_idx = 0;
    following_layer.weights.indexed_iter().for_each(
        |((row_idx, col_idx), weight)| if *weight != zero_component {
            new_weights[[connection_idx, col_idx]] = T::Component::one();
            following_weights[[row_idx, connection_idx]] = *weight;
            connection_idx += 1;
        }
    );
    following_layer.weights = following_weights;
    let new_biases = T::zeros([new_weights_num_rows, 1]);
    let new_layer = Layer::new(new_weights, new_biases, following_layer.activation.clone());
    layers.insert(new_layer_idx, new_layer);
}


fn should_add_layer(num_layers: usize, rng: &mut ThreadRng) -> bool {
    let threshold = 1.0/2.0_f32.powi((num_layers - 1) as i32);
    rng.gen_range(0.0..1.0) < threshold
}


/// Randomly adds new connections between existing neurons.
pub fn mutate_add_connections<T: Tensor2>(layers: &mut [Layer<T>], rng: &mut ThreadRng) {
    let neuron_index_lookup = get_neuron_index_lookup(layers);
    let total_size = neuron_index_lookup.len();
    // Decide how many new connections to create.
    let dist = Poisson::new(1.).unwrap();
    let num_new_connections = dist.sample(rng) as usize;
    (0..num_new_connections).for_each(|_| add_new_connection(layers, total_size, &neuron_index_lookup, rng));
}


/// Returns a vector that maps "global" neuron index to 2-tuple of layer index and "in-layer" neuron index.
fn get_neuron_index_lookup<T: Tensor2>(layers: &[Layer<T>]) -> Vec<(usize, usize)> {
    let size = layers.iter().map(|layer| layer.size()).sum();
    let mut neuron_index_lookup = Vec::with_capacity(size);
    for (layer_idx, layer) in layers.iter().enumerate() {
        for neuron_idx in 0..layer.size() {
            neuron_index_lookup.push((layer_idx, neuron_idx))
        }
    }
    neuron_index_lookup
}


/// Randomly chooses a start and end of a new connection.
fn choose_new_connection<T: Tensor2>(
    layers: &[Layer<T>],
    total_size: usize,
    neuron_index_lookup: &[(usize, usize)],
    rng: &mut ThreadRng,
) -> (usize, usize, usize, usize) {
    let global_idx = rng.gen_range(0..total_size);
    let (mut start_layer_idx, mut start_neuron_idx) = neuron_index_lookup[global_idx];
    let distribution_weights: Vec<f32> = layers.iter().enumerate().map(
        |(layer_idx, layer)| {
            let distance = (layer_idx as isize - start_layer_idx as isize).abs() as i32; // cast could theoretically lead to wraparound if there is an absurd amount of layers
            if distance == 0 { 0. } else { layer.size() as f32 / 2f32.powi(distance - 1) }
        }
    ).collect();
    let dist = WeightedIndex::new(&distribution_weights).unwrap();
    let mut end_layer_idx = dist.sample(rng);
    let mut end_neuron_idx = rng.gen_range(0..layers[end_layer_idx].size());
    if end_layer_idx < start_layer_idx {
        (end_layer_idx, end_neuron_idx, start_layer_idx, start_neuron_idx) = (start_layer_idx, start_neuron_idx, end_layer_idx, end_neuron_idx);
    }
    (start_layer_idx, start_neuron_idx, end_layer_idx, end_neuron_idx)
}


/// Creates new connections and possibly neurons in between.
fn add_new_connection<T: Tensor2>(
    layers: &mut [Layer<T>],
    total_size: usize,
    neuron_index_lookup: &[(usize, usize)],
    rng: &mut ThreadRng,
) {
    let (start_layer_idx, start_neuron_idx, end_layer_idx, end_neuron_idx) = choose_new_connection(layers, total_size, neuron_index_lookup, rng);
    let mut prev_neuron_idx = start_neuron_idx;
    for layer_idx in (start_layer_idx + 1)..end_layer_idx {
        // Append a row to weight matrix of intermediate layer.
        // The row is all zeros except for the column corresponding to the neuron connected to
        // it from the previous layer.
        let weights = &mut layers[layer_idx].weights;
        let (num_rows, num_cols) = weights.shape();
        let mut new_row = vec![T::Component::zero(); num_cols];
        new_row[prev_neuron_idx] = T::Component::one();
        weights.append(0, &new_row);
        // Remember the index of the new neuron for the next iteration.
        prev_neuron_idx = num_rows;
        // Append a zero to the bias vector.
        layers[layer_idx].biases.append(0, &vec![T::Component::zero()]);
        // Append a column to the next layer's weight matrix with all zeros.
        let weights = &mut layers[layer_idx + 1].weights;
        let (num_rows, _) = weights.shape();
        weights.append(1, &vec![T::Component::zero(); num_rows]);
    }
    layers[end_layer_idx].weights[[end_neuron_idx, prev_neuron_idx]] = init_weight(rng, false).unwrap();
}


