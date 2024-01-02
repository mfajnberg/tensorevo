// TODO: Documentation
//       https://github.com/mfajnberg/tensorevo/issues/22

use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::tensor::Tensor;


#[derive(Clone, Deserialize, Serialize, PartialEq, Debug)]
#[serde(bound = "")]
pub struct Layer<T: Tensor> {
    pub weights: T,
    pub biases: T,
    pub activation: Activation<T>
}

impl<T: Tensor> Layer<T> {
    pub fn new(weights: T, biases: T, activation: Activation<T>) -> Self {
        Self { weights, biases, activation }
    }

    pub fn feed_forward(&self, input: &T) -> (T, T) {
        let weighted_input = &self.weights.dot(input) + &self.biases;
        let activation = (self.activation)(&weighted_input);
        (weighted_input, activation)
    }
}


impl<T: Tensor> FnOnce<(&T,)> for Layer<T> {
    type Output = T;

    extern "rust-call" fn call_once(self, args: (&T,)) -> Self::Output {
        self.feed_forward(args.0).1
    }
}


impl<T: Tensor> FnMut<(&T,)> for Layer<T> {
    extern "rust-call" fn call_mut(&mut self, args: (&T,)) -> Self::Output {
        self.feed_forward(args.0).1
    }
}


impl<T: Tensor> Fn<(&T,)> for Layer<T> {
    extern "rust-call" fn call(&self, args: (&T,)) -> Self::Output {
        self.feed_forward(args.0).1
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};
    use num_traits::One;

    use super::*;

    fn double<T: Tensor>(t: &T) -> T {
        t + t
    }

    fn twos<T: Tensor>(t: &T) -> T {
        let ones = T::from_num(T::Component::one(), t.shape());
        &ones + &ones
    }

    #[test]
    fn test_new() {
        let weights = array![[1., 0.], [0., 1.]];
        let biases = array![[0.], [0.]];
        let activation = Activation::new("double", double, twos);

        let layer = Layer::<Array2<f32>>::new(weights.clone(), biases.clone(), activation.clone());
        assert_eq!(layer, Layer { weights, biases, activation });
    }

    #[test]
    fn test_feed_forward() {
        let weights = array![[1., 0.], [0., 1.]];
        let biases = array![[1.], [1.]];
        let activation = Activation::new("double", double, twos);
        let layer = Layer { weights, biases, activation };

        let input = array![[0.], [42.]];
        let (weighted_input, output) = layer.feed_forward(&input);
        assert_eq!(weighted_input, array![[1.], [43.]]);
        assert_eq!(output, array![[2.], [86.]]);
    }

    # [test]
    fn test_fn_traits() {
        let weights = array![[1., 0.], [0., 1.]];
        let biases = array![[1.], [1.]];
        let activation = Activation::new("double", double, twos);
        let mut layer = Layer { weights, biases, activation };

        let input = array![[0.], [1.]];
        let expected_output = array![[2.], [4.]];

        // Receiver borrowed:
        let output = layer(&input);
        assert_eq!(output, expected_output);

        // Receiver mutably borrowed:
        let output = layer.call_mut((&input,));
        assert_eq!(output, expected_output);

        // Receiver moved:
        let output = layer.call_once((&input,));
        assert_eq!(output, expected_output);
    }
}
