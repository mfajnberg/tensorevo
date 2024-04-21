use ndarray::{Array2, Axis, array, stack};

use tensorevo::activation::{Activation, Registered};
use tensorevo::cost_function::CostFunction;
use tensorevo::individual::Individual;
use tensorevo::layer::Layer;

#[test]
fn test_sgd() {
    let mut individual = Individual::new(
        vec![
            Layer{
                weights: array![
                    [1., 0.],
                    [0., 1.],
                ],
                biases: array![
                    [0.],
                    [0.],
                ],
                activation: Activation::get("relu").unwrap(),
            },
            Layer{
                weights: array![
                    [1., 0.],
                    [0., 1.],
                ],
                biases: array![
                    [0.],
                    [0.],
                ],
                activation: Activation::get("relu").unwrap(),
            },
        ],
        CostFunction::<Array2<f64>>::get("quadratic").unwrap(),
    );
    let input_1 = array![
        [1., 2.],
        [1., 3.],
    ];
    let input_2 = array![
        [4., 1.],
        [5., 1.],
    ];
    let desired_output_1 = array![
        [10., 5.],
        [1., 5.],
    ];
    let desired_output_2 = array![
        [5., 10.],
        [4., 1.],
    ];
    let input_batches = stack(Axis(0), &[input_1.view(), input_2.view()]).unwrap();
    let desired_output_batches = stack(Axis(0), &[desired_output_1.view(), desired_output_2.view()]).unwrap();

    let error_pre_training_1 = individual.calculate_error(&input_1, &desired_output_1);
    let error_pre_training_2 = individual.calculate_error(&input_2, &desired_output_2);
    for _ in 0..100 {
        individual.stochastic_gradient_descent((&input_batches, &desired_output_batches), 0.01, None);
    }
    let error_post_training_1 = individual.calculate_error(&input_1, &desired_output_1);
    let error_post_training_2 = individual.calculate_error(&input_2, &desired_output_2);

    assert!(
        error_post_training_1 < error_pre_training_1,
        "Training error did not decrease.\nBefore: {error_pre_training_1}. After: {error_post_training_1}",
    );
    assert!(
        error_post_training_2 < error_pre_training_2,
        "Training error did not decrease.\nBefore: {error_pre_training_2}. After: {error_post_training_2}",
    );
}

