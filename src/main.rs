mod data;
mod model;
mod training;

use anyhow::Result;
use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    module::{AutodiffModule, Module},
};

// Hyperparameters
const SEED: u64 = 42;
const TRAIN_RATIO: f64 = 0.7;
const LEARNING_RATE: f64 = 0.0001;
const WEIGHT_DECAY: f64 = 0.0001;
const EPOCHS: usize = 10000;
const REPORT_INTERVAL: usize = 1000;

fn main() -> Result<()> {
    println!("-----------------------");
    println!("data.rs:");
    println!("-----------------------");

    let wine_df_raw = data::fetch_data()?;
    let wine_df = data::shuffle_data(wine_df_raw, Some(SEED))?;

    let (features, labels) = data::split_xy(wine_df)?;

    let normalized_features = data::normalize(features)?;
    let normalized_features_tensor = data::df_to_tensor::<NdArray>(normalized_features)?;

    let labels_tensor = data::series_to_tensor::<NdArray>(labels)?;

    println!(
        "Features tensor shape: {:?}",
        normalized_features_tensor.shape()
    );
    println!("Labels tensor shape: {:?}", labels_tensor.shape());

    let (x_train, y_train, x_test, y_test) =
        data::train_test_split(normalized_features_tensor, labels_tensor, TRAIN_RATIO);

    println!(
        "x_train shape: {:?}, y_train shape: {:?}",
        x_train.shape(),
        y_train.shape()
    );
    println!(
        "x_test shape: {:?}, y_test shape: {:?}",
        x_test.shape(),
        y_test.shape()
    );

    println!();
    println!("-----------------------");
    println!("model.rs:");
    println!("-----------------------");

    let device = NdArrayDevice::Cpu;
    let model = model::SimpleClassifier::<NdArray>::new(&device);

    // println!("Model structure: {:#?}", model);
    println!("Total parameters: {}", model.num_params());

    let output = model.forward(x_train.clone());
    println!("Output shape: {:?}", output.shape());

    println!();
    println!("-----------------------");
    println!("training.rs:");
    println!("-----------------------");

    let model = model::SimpleClassifier::<Autodiff<NdArray>>::new(&device);

    let initial_validation_model = (&model).valid();

    let initial_train_output = initial_validation_model.forward(x_train.clone());
    let initial_test_output = initial_validation_model.forward(x_test.clone());

    let initial_trian_accuracy =
        training::calculate_accuracy(initial_train_output.clone(), y_train.clone());
    let initial_test_accuracy =
        training::calculate_accuracy(initial_test_output.clone(), y_test.clone());

    println!(
        "Model accuracy using train dataset before training: {}",
        initial_trian_accuracy
    );
    println!(
        "Model accuracy using test dataset before training: {}",
        initial_test_accuracy
    );
    println!();
    println!("----------- Training reports -----------");

    let (trained_model, train_losses, test_losses) = training::train(
        model,
        x_train.clone(),
        y_train.clone(),
        x_test.clone(),
        y_test.clone(),
        WEIGHT_DECAY,
        EPOCHS,
        REPORT_INTERVAL,
        LEARNING_RATE,
    );

    println!();
    println!("Final train loss: {:.4}", train_losses.last().unwrap());
    println!("Final test loss: {:.4}", test_losses.last().unwrap());

    let final_validation_model = (&trained_model).valid();

    let final_train_output = final_validation_model.forward(x_train.clone());
    let final_test_output = final_validation_model.forward(x_test.clone());

    let final_train_accuracy =
        training::calculate_accuracy(final_train_output.clone(), y_train.clone());
    let final_test_accuracy =
        training::calculate_accuracy(final_test_output.clone(), y_test.clone());

    println!(
        "Model accuracy using train dataset after training: {:.4}",
        final_train_accuracy
    );
    println!(
        "Model accuracy using test dataset after training: {:.4}",
        final_test_accuracy
    );

    // println!("{:?}", train_losses);
    // println!("{:?}", test_losses);

    Ok(())
}
