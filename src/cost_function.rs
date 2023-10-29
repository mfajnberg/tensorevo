use num_traits::ToPrimitive;

use crate::tensor::Tensor;


/// Reference implementation of the quadratic cost function.
///
/// Calculated as the euclidian norm of the difference between actual and desired output divided by
/// two.
/// Used primarily during stochastic gradient descent.
///
/// # Arguments
/// * `output` - The actual output returned by a individual as a tensor.
/// * `desired_output` - The desired output as a tensor.
///
/// # Returns
/// The cost as 32 bit float.
pub fn quadratic_cost_function<T: Tensor>(
    output: &T,
    desired_output: &T,
) -> f32 {
    return desired_output.sub(output).vec_norm().to_f32().unwrap() / 2.
}


/// Reference implementation of the derivative of the quadratic cost function.
///
/// Simply the difference between actual and desired output.
/// Used only during stochastic gradient descent.
///
/// # Arguments
/// * `output` - The actual output returned by a individual as a tensor.
/// * `desired_output` - The desired output as a tensor.
///
/// # Returns
/// Another tensor.
pub fn quadratic_cost_function_prime<T: Tensor>(
    output: &T,
    desired_output: &T,
) -> T {
    return output.sub(desired_output)
}
