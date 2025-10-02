mod data;

use anyhow::Result;
use burn::backend::NdArray;

// Hyperparameters
const SEED: u64 = 42;
const TRAIN_RATIO: f32 = 0.7;

fn main() -> Result<()> {
    println!("Data unit:");

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

    Ok(())
}
