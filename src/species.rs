use crate::individual::Individual;
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
}


impl<T: Tensor> Default for Species<T> {
    fn default() -> Self {
        Species::new()
    }
}
