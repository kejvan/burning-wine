//! Data processing utilities for wine classification
//!
//! This module provides functions for downloading, loading, preprocessing,
//! and converting wine dataset into Burn tensors for training.

use anyhow::Result;
use burn::tensor::{Float, Int, Tensor, backend::Backend};
use polars::prelude::*;
use std::fs;
use std::path::Path;

/// Fetches the wine dataset from UCI repository or loads from local cache
///
/// Downloads the dataset if not present at `./data/wine_dataset.csv`,
/// otherwise loads the existing file.
///
/// # Returns
/// * `Ok(DataFrame)` - The wine dataset with auto-generated column names
/// * `Err` - If download or file reading fails
///
/// # Example
/// ```
/// let wine_df = fetch_data()?;
/// ```
pub fn fetch_data() -> Result<DataFrame> {
    let path = "./data/wine_dataset.csv";
    let url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data";
    if !Path::new(path).exists() {
        println!("Downloading data to {}...", path);
        fs::create_dir_all("./data")?;

        let response = reqwest::blocking::get(url)?;
        let content = response.text()?;
        fs::write(path, content)?;
        println!("Download complete.");
    } else {
        println!("Loading existing file at {}", path)
    }

    let file = fs::File::open(&path)?;
    let mut wine_df = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(file)
        .finish()?;

    // Create column names
    let col_names: Vec<_> = (0..wine_df.width())
        .map(|i| format!("column_{}", i))
        .collect();

    // Set column names
    wine_df.set_column_names(&col_names)?;

    Ok(wine_df)
}

/// Randomly shuffles the rows of a DataFrame
///
/// # Arguments
/// * `df` - The DataFrame to shuffle
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `Ok(DataFrame)` - Shuffled DataFrame with same shape
/// * `Err` - If shuffling fails
///
/// # Example
/// ```
/// let shuffled = shuffle_data(df, Some(42))?;  // Reproducible
/// let shuffled = shuffle_data(df, None)?;      // Random
/// ```
pub fn shuffle_data(df: DataFrame, seed: Option<u64>) -> Result<DataFrame> {
    let num_rows = df.height();
    let shuffled = df.sample_n_literal(num_rows, false, true, seed)?;

    Ok(shuffled)
}

/// Splits DataFrame into features (x) and labels (y)
///
/// Separates the first column (class labels) from the remaining columns (features).
///
/// # Arguments
/// * `df` - The complete wine dataset DataFrame
///
/// # Returns
/// * `Ok((DataFrame, Series))` - Tuple of (features, labels)
///   - features: columns 1-13 (all feature columns)
///   - labels: column 0 (wine class)
/// * `Err` - If splitting fails
pub fn split_xy(df: DataFrame) -> Result<(DataFrame, Series)> {
    // Separate labels (column 0 = class)
    let y = df
        .select_at_idx(0)
        .unwrap()
        .as_materialized_series()
        .clone();

    // Extract features (columns 1-13)
    let x = df.select_by_range(1..df.width())?;

    Ok((x, y))
}

/// Normalizes features using StandardScaler (mean=0, std=1)
///
/// Applies z-score normalization to each column: (x - mean) / std
/// Uses sample standard deviation (ddof=1) like sklearn's StandardScaler.
///
/// # Arguments
/// * `df` - DataFrame with feature columns to normalize
///
/// # Returns
/// * `Ok(DataFrame)` - Normalized DataFrame with same shape
/// * `Err` - If normalization fails
pub fn normalize(df: DataFrame) -> Result<DataFrame> {
    // Normalize each column: (x - mean) / std
    let normalized_columns: Vec<_> = df
        .iter()
        .map(|col| {
            // Cast to f32 for calculations
            let series = col.cast(&DataType::Float32).unwrap();
            let mean = series.mean().unwrap();
            let std = series.std(1).unwrap(); // ddof=1 for sample std

            // Apply z-score normalization
            let normalized_series = (&series - mean) / std;
            normalized_series.into_column()
        })
        .collect();

    DataFrame::new(normalized_columns).map_err(|e| e.into())
}

/// Converts a Polars Series of labels into a 1D Burn tensor
///
/// Transforms wine class labels (1, 2, 3) into zero-indexed classes (0, 1, 2)
/// suitable for neural network training.
///
/// # Arguments
/// * `series` - Polars Series containing class labels
///
/// # Returns
/// * `Ok(Tensor<B, 1>)` - 1D integer tensor of class indices
/// * `Err` - If conversion fails
pub fn series_to_tensor<B: Backend>(series: Series) -> Result<Tensor<B, 1, Int>> {
    // Convert series to vector and shift classes from 1-3 to 0-2
    let series_vec: Vec<i64> = series
        .i64()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap_or(0) - 1) // Convert 1,2,3 -> 0,1,2
        .collect();

    // Create Burn tensor from vector
    let series_tensor =
        Tensor::<B, 1, Int>::from_ints(series_vec.as_slice(), &B::Device::default());

    Ok(series_tensor)
}

/// Converts a Polars DataFrame into a 2D Burn tensor
///
/// Transforms feature DataFrame into a 2D float tensor suitable for
/// neural network input.
///
/// # Arguments
/// * `df` - Polars DataFrame containing feature columns
///
/// # Returns
/// * `Ok(Tensor<B, 2>)` - 2D float tensor of shape [num_rows, num_features]
/// * `Err` - If conversion fails
pub fn df_to_tensor<B: Backend>(df: DataFrame) -> Result<Tensor<B, 2, Float>> {
    // Convert each column to f32 vector
    let df_vec: Vec<Vec<f32>> = df
        .iter()
        .map(|series| {
            series
                .cast(&DataType::Float32)
                .unwrap()
                .f32()
                .unwrap()
                .into_iter()
                .map(|v| v.unwrap_or(0.0) as f32)
                .collect()
        })
        .collect();

    // Get dimensions and flatten to 1D for tensor creation
    let num_rows = df.height();
    let num_cols = df.width();
    let df_flat: Vec<f32> = df_vec.into_iter().flatten().collect();

    // Create 2D Burn tensor
    let df_tensor = Tensor::<B, 1, Float>::from_floats(df_flat.as_slice(), &B::Device::default())
        .reshape([num_rows, num_cols]);

    Ok(df_tensor)
}

/// Splits tensors into training and test sets
///
/// Performs a simple train/test split based on the given ratio.
/// Note: Data should be shuffled before calling this function.
///
/// # Arguments
/// * `x` - 2D feature tensor of shape [num_samples, num_features]
/// * `y` - 1D label tensor of shape [num_samples]
/// * `train_ratio` - Proportion of data for training (e.g., 0.7 for 70%)
///
/// # Returns
/// Tuple of (x_train, y_train, x_test, y_test)
/// * `x_train` - Training features
/// * `y_train` - Training labels
/// * `x_test` - Test features
/// * `y_test` - Test labels
pub fn train_test_split<B: Backend>(
    x: Tensor<B, 2, Float>,
    y: Tensor<B, 1, Int>,
    train_ratio: f64,
) -> (
    Tensor<B, 2, Float>,
    Tensor<B, 1, Int>,
    Tensor<B, 2, Float>,
    Tensor<B, 1, Int>,
) {
    let total_samples = x.dims()[0];
    let train_size = (total_samples as f64 * train_ratio) as usize;
    let num_cols = x.dims()[1];

    // Split features into train and test
    let x_train = x.clone().slice([0..train_size, 0..num_cols]);
    let x_test = x.slice([train_size..total_samples, 0..num_cols]);

    // Split labels into train and test
    let y_train = y.clone().slice([0..train_size]);
    let y_test = y.slice([train_size..total_samples]);

    (x_train, y_train, x_test, y_test)
}
