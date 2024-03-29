use crate::individual::Individual;
use crate::tensor::Tensor;


pub fn get_species<T: Tensor>(individual: &Individual<T>) -> String {
    individual.num_layers().to_string()
}

