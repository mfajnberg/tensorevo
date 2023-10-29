use crate::individual::Individual;
use crate::tensor::Tensor;

pub struct Species<T: Tensor> 
    where
        T: Tensor,
        T::Element: From<f32>,
{
    individuals: Vec<Individual<T>>
}
impl<T: Tensor> Species<T> 
    where
        T: Tensor,
        T::Element: From<f32>,
{
    pub fn new() -> Self {
        return Self {
            individuals: Vec::new()
        }
    }
    pub fn add(&mut self, individual: Individual<T>) {
        self.individuals.push(individual);
    }
}