use ndarray::{Array2, array};

use tensorevo::activation::Activation;
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
                activation: Activation::from_name("relu"),
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
                activation: Activation::from_name("relu"),
            },
        ],
        CostFunction::<Array2<f64>>::from_name("quadratic"),
    );
    let input1 = array![
        [1., 2.],
        [1., 3.],
    ];
    let input2 = array![
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
    let training_data = vec![
        (&input1, &desired_output_1),
        (&input2, &desired_output_2),
    ];

    let error_pre_training_1 = individual.calculate_error(&input1, &desired_output_1);
    let error_pre_training_2 = individual.calculate_error(&input2, &desired_output_2);
    for _ in 0..100 {
        individual.stochastic_gradient_descent(training_data.clone(), 0.01, None);
    }
    let error_post_training_1 = individual.calculate_error(&input1, &desired_output_1);
    let error_post_training_2 = individual.calculate_error(&input2, &desired_output_2);

    assert!(
        error_post_training_1 < error_pre_training_1,
        "Training error did not decrease.\nBefore: {}. After: {}",
        error_pre_training_1,
        error_post_training_1,
    );
    assert!(
        error_post_training_2 < error_pre_training_2,
        "Training error did not decrease.\nBefore: {}. After: {}",
        error_pre_training_2,
        error_post_training_2,
    );
}

