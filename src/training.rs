//! Training utilities for the wine classifier neural network
//!
//! This module provides functions for training and evaluating the model:
//! - Training loop with Adam optimizer, cosine learning rate scheduling, and weight decay
//! - Accuracy calculation for evaluation
//! - Loss tracking for both training and test sets

use crate::model::SimpleClassifier;
use burn::{
    lr_scheduler::{LrScheduler, cosine::CosineAnnealingLrSchedulerConfig},
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer, decay::WeightDecayConfig},
    tensor::{
        ElementConversion, Float, Int, Tensor,
        backend::{AutodiffBackend, Backend},
    },
};

/// Calculates classification accuracy
///
/// Compares predicted class indices with true labels and returns
/// the accuracy as a fraction (0.0 to 1.0).
///
/// # Arguments
/// * `outputs` - Model output logits of shape [batch_size, num_classes]
/// * `targets` - True class labels of shape [batch_size]
///
/// # Returns
/// * `f32` - Accuracy as a fraction (0.0 to 1.0)
///
/// # Example
/// ```
/// let output = model.forward(x_train);
/// let accuracy = calculate_accuracy(output, y_train);
/// println!("Accuracy: {:.2}%", accuracy * 100.0);
/// ```
pub fn calculate_accuracy<B: Backend>(
    outputs: Tensor<B, 2, Float>,
    targets: Tensor<B, 1, Int>,
) -> f32 {
    let predictions: Tensor<B, 1, Int> = outputs.argmax(1).squeeze(1);
    let total = targets.dims()[0] as f32;
    let correct: f32 = predictions
        .equal(targets)
        .float()
        .sum()
        .into_scalar()
        .elem();
    correct / total
}

/// Trains the wine classifier model
///
/// Executes the training loop for the specified number of epochs using:
/// - Adam optimizer with configurable weight decay (L2 regularization)
/// - Cosine annealing learning rate scheduler
/// - CrossEntropyLoss
/// - Progress logging at specified intervals
///
/// The function converts input tensors from the inner backend to autodiff backend
/// for training, then validates on the inner backend (without autodiff overhead)
/// each epoch to track test loss.
///
/// # Arguments
/// * `model` - The classifier model to train (on autodiff backend)
/// * `x_train` - Training features of shape [n_samples, 13] (inner backend)
/// * `y_train` - Training labels of shape [n_samples] (inner backend)
/// * `x_test` - Test features of shape [n_samples, 13] (inner backend)
/// * `y_test` - Test labels of shape [n_samples] (inner backend)
/// * `weight_decay` - L2 regularization penalty (e.g., 0.0001)
/// * `epochs` - Number of training epochs
/// * `report_interval` - Print progress every N epochs
/// * `learning_rate` - Initial learning rate for optimizer
///
/// # Returns
/// * `(SimpleClassifier<B>, Vec<f64>, Vec<f64>)` - Tuple containing:
///   - Trained model (on autodiff backend)
///   - Training loss history (one value per epoch)
///   - Test loss history (one value per epoch)
///
/// # Example
/// ```
/// let (trained_model, train_losses, test_losses) = train(
///     model,
///     x_train,
///     y_train,
///     x_test,
///     y_test,
///     0.0001,  // weight_decay
///     1000,    // epochs
///     200,     // report_interval
///     0.001,   // learning_rate
/// );
/// ```
pub fn train<B: AutodiffBackend>(
    mut model: SimpleClassifier<B>,
    x_train: Tensor<B::InnerBackend, 2, Float>,
    y_train: Tensor<B::InnerBackend, 1, Int>,
    x_test: Tensor<B::InnerBackend, 2, Float>,
    y_test: Tensor<B::InnerBackend, 1, Int>,
    weight_decay: f64,
    epochs: usize,
    report_interval: usize,
    learning_rate: f64,
) -> (SimpleClassifier<B>, Vec<f64>, Vec<f64>) {
    // Convert tensors from inner backend to autodiff backend
    let x_train = Tensor::<B, 2, Float>::from_inner(x_train);
    let y_train = Tensor::<B, 1, Int>::from_inner(y_train);

    let train_device: &<B as Backend>::Device = &x_train.device();
    let train_criterion = CrossEntropyLossConfig::new().init(train_device);

    let weight_decay = WeightDecayConfig {
        penalty: weight_decay as f32,
    };

    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(weight_decay))
        .init::<B, SimpleClassifier<B>>();

    let scheduler_config = CosineAnnealingLrSchedulerConfig::new(learning_rate, epochs);
    let mut lr_scheduler = scheduler_config.init().unwrap();

    let mut train_loss_history: Vec<f64> = vec![];
    let mut test_loss_history: Vec<f64> = vec![];

    for epoch in 1..=epochs {
        let train_output = model.forward(x_train.clone());

        let train_loss = train_criterion.forward(train_output.clone(), y_train.clone());
        let train_loss_value: f64 = train_loss.clone().into_scalar().elem();
        train_loss_history.push(train_loss_value);

        let grads = train_loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        let current_lr = lr_scheduler.step();
        model = optimizer.step(current_lr, model, grads);

        let validation_model = (&model).valid();
        let test_output = validation_model.forward(x_test.clone());

        let test_device = &x_test.device();
        let test_criterion = CrossEntropyLossConfig::new().init(test_device);

        let test_loss = test_criterion.forward(test_output.clone(), y_test.clone());
        let test_loss_value: f64 = test_loss.clone().into_scalar().elem();
        test_loss_history.push(test_loss_value);

        if epoch % report_interval == 0 || epoch == 1 {
            println!(
                "Train loss: {:.4}, test loss: {:.4}",
                train_loss_value, test_loss_value
            );
        }
    }

    (model, train_loss_history, test_loss_history)
}
