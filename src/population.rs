use std::cmp::{min, max};
use std::collections::HashMap;

use ordered_float::OrderedFloat;
use rand::distributions::Slice;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::{Rng, thread_rng};

use crate::individual::Individual;
use crate::layer::Layer;
use crate::tensor::Tensor;


type SelectFunc<T> = fn(&mut Population<T>, &str) -> Vec<(usize, usize)>;
type ProcreateFunc<T> = fn(&Individual<T>, &Individual<T>) -> Individual<T>;
type DetermineSpeciesKey<T> = fn(&Individual<T>) -> String;


// TODO: The term "population" refers to a group of organisms of the **same** species.
//       We should find a better term.
//       It seems the term "community" or even "environment" might be suitable.
pub struct Population<T: Tensor> {
    pub species: HashMap<String, Vec<Individual<T>>>,
    kill_weak_and_select_parents: SelectFunc<T>,
    procreate_pair: ProcreateFunc<T>, // crossover, mutation
    determine_species_key: DetermineSpeciesKey<T>,
    pub validation_data: (T, T),
}


impl<T: Tensor> Population<T> {
    pub fn new(
        kill_weak_and_select_parents: SelectFunc<T>,
        procreate_pair: ProcreateFunc<T>,
        determine_species_key: DetermineSpeciesKey<T>,
        validation_data: (T, T),
    ) -> Self {
        Self {
            species: HashMap::new(),
            kill_weak_and_select_parents,
            procreate_pair,
            determine_species_key,
            validation_data,
        }
    }
    
    // mutates sets of individuals within one species
    pub fn apply_selection(&mut self, species_key: String) {
        let pairs = (self.kill_weak_and_select_parents)(self, &species_key);
        let species = self.species.get(&species_key).unwrap();
        let new_individuals: Vec<Individual<T>> = pairs.into_iter().map(
            |(index1, index2)| (self.procreate_pair)(&species[index1], &species[index2])
        ).collect();
        for new_individual in new_individuals {
            self.add_new_individual(new_individual)
        }
    }

    pub fn add_new_individual(&mut self, individual: Individual<T>) {
        let key = (self.determine_species_key)(&individual);
        match self.species.get_mut(&key) {
            Some(existing_species) => existing_species.push(individual),
            None => {
                self.species.insert(key, vec![individual]);
            }
        }
    }
}


/// Reference implementation of a `kill_weak_and_select_parents` function.
///
/// Calculates validation error for all individuals in the specifided species,
/// kills half of them (those with the highest errors), randomly samples parents
/// from the remaining individuals (using a uniform distribution with replacement)
/// and returns their indices as 2-tuples.
///
/// # Arguments:
/// `population`  - The population containing the species select from.
/// `species_key` - The key of the species to select from.
///
/// # Returns:
/// Vector of 2-tuples of indices of the individuals in the specified species that should procreate.
pub fn select<T: Tensor>(population: &mut Population<T>, species_key: &str) -> Vec<(usize, usize)> {
    // Sort individuals in species by validation error.
    let individuals = population.species.get_mut(species_key).unwrap();
    let (input, desired_output) = &population.validation_data;
    individuals.sort_by_cached_key(
        |individual| OrderedFloat(individual.calculate_error(input, desired_output))
    );
    // Kill the less fit half.
    let num_individuals = individuals.len();
    let num_killed = individuals.drain(num_individuals/2..num_individuals).count();
    // Randomly sample (with a uniform distribution, with replacement) twice as many parents.
    // If we wanted the probability to become a parent to depend on the validation error,
    // we could use `rand::distributions::weighted::WeightedIndex` instead.
    let rng = thread_rng();
    let indices = (0..(num_individuals - num_killed)).collect::<Vec<usize>>();
    let index_distribution = Slice::new(&indices).unwrap();
    rng.sample_iter(&index_distribution)
       .take(num_killed * 2)
       .array_chunks::<2>()
       .map(|chunk| (*chunk[0], *chunk[1]))
       .collect()
}



fn crossover_layer_weights<T: Tensor>(
    new_weights: &mut T,
    weights_p1: &T,
    weights_p2: &T,
    rng: &mut ThreadRng,
) {
    let (rows_p1, cols_p1) = weights_p1.shape();
    let (rows_p2, cols_p2) = weights_p2.shape();
    for ((row, col), component) in new_weights.indexed_iter_mut() {
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
            // TODO: maybe initialize new component
        }
    }
}


fn crossover_layer_biases<T: Tensor>(
    new_biases: &mut T,
    biases_p1: &T,
    biases_p2: &T,
    rng: &mut ThreadRng,
) {
    let rows_p1 = biases_p1.shape().0;
    let rows_p2 = biases_p2.shape().0;
    for ((row, _), component) in new_biases.indexed_iter_mut() {
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


// TODO: Factor out lots of things
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
    Individual::new(layers, parent_1.get_cost_function().clone())
}


pub fn get_species<T: Tensor>(individual: &Individual<T>) -> String {
    individual.num_layers().to_string()
}


#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_new() {
        let _pop = Population::new(
            select,
            procreate,
            get_species,
            (array![[0.]], array![[0.]]),
        );
    }
}
