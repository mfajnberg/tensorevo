use ordered_float::OrderedFloat;

use crate::individual::{Individual, self};
use crate::tensor::Tensor;


pub struct Species<T: Tensor> {
    individuals: Vec<Individual<T>>
}


impl<T: Tensor> Species<T> {
    pub fn new() -> Self {
        Self {
            individuals: Vec::new()
        }
    }

    pub fn add(&mut self, individual: Individual<T>) {
        self.individuals.push(individual);
    }
    
    pub fn sort_by_error(&mut self, input: &T, desired_output: &T) {
        self.individuals.sort_by_cached_key(
            |individual| OrderedFloat(individual.calculate_error(input, desired_output))
        );
    }

    pub fn size(&self) -> usize {
        return self.individuals.len();
    }

    pub fn drain_individuals(&mut self, start: usize, end: usize) {
        self.individuals.drain(start..end);
    }
    
}


impl<T: Tensor> Default for Species<T> {
    fn default() -> Self {
        Species::new()
    }
}