//! Neural network model for wine classification
//!
//! This module provides a simple feed-forward neural network classifier
//! for the UCI wine dataset with 3 output classes.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Float, Tensor, backend::Backend},
};

/// Neural network model for wine classification
///
/// Architecture: 13 → 4 (ReLU) → 4 (ReLU) → 3
///
/// The model consists of:
/// - Input layer: 13 features (chemical properties)
/// - Hidden layer 1: 4 neurons with ReLU activation
/// - Hidden layer 2: 4 neurons with ReLU activation
/// - Output layer: 3 neurons (logits for classes 0, 1, 2)
///
/// # Example
/// ```
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
///
/// let device = NdArrayDevice::Cpu;
/// let model = SimpleClassifier::<NdArray>::new(&device);
/// let output = model.forward(input_tensor);
/// ```
#[derive(Module, Debug)]
pub struct SimpleClassifier<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    activation: Relu,
}

impl<B: Backend> SimpleClassifier<B> {
    /// Creates a new wine classifier model
    ///
    /// Initializes all layers with random weights using the default
    /// initialization strategy.
    ///
    /// # Arguments
    /// * `device` - The device to initialize the model on (CPU/GPU)
    ///
    /// # Returns
    /// * `Self` - A new instance with randomly initialized weights
    ///
    /// # Example
    /// ```
    /// let device = NdArrayDevice::Cpu;
    /// let model = SimpleClassifier::<NdArray>::new(&device);
    /// ```
    pub fn new(device: &B::Device) -> Self {
        let linear1: Linear<B> = LinearConfig::new(13, 4).init(device);
        let linear2: Linear<B> = LinearConfig::new(4, 4).init(device);
        let linear3: Linear<B> = LinearConfig::new(4, 3).init(device);
        let activation = Relu::new();

        Self {
            linear1,
            linear2,
            linear3,
            activation,
        }
    }

    /// Performs a forward pass through the network
    ///
    /// Computes predictions by passing input through all layers:
    /// input → linear1 → relu → linear2 → relu → linear3 → logits
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, 13]
    ///
    /// # Returns
    /// * `Tensor<B, 2, Float>` - Output logits of shape [batch_size, 3]
    ///
    /// # Example
    /// ```
    /// let output = model.forward(x_train);
    /// let predictions = output.argmax(1);
    /// ```
    pub fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        self.linear3.forward(x)
    }
}
