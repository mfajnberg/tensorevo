use crate::individual::Individual;
use crate::tensor::Tensor;


pub struct Species<T>
where
    T: Tensor,
    T::Element: From<f32>,
{
    individuals: Vec<Individual<T>>
}


impl<T> Species<T>
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


impl<T> Default for Species<T>
where
    T: Tensor,
    T::Element: From<f32>,
{
    fn default() -> Self {
        return Species::new();
    }
}
