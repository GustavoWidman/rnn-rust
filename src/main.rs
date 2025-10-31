mod data;
mod model;
mod utils;

use clap::Parser;
use eyre::Result;
use log::info;

use crate::data::Dataset;
use crate::model::RNN;
use crate::model::activation::ActivationFunction;
#[allow(unused_imports)]
use crate::model::evaluator::{BinaryClassificationEvaluator, RegressionEvaluator};
use crate::model::loss::LossFunction;
use crate::model::optimizer::{Optimizer, SGD};
use crate::utils::cli::Args;
use crate::utils::log::Logger;

fn main() -> Result<()> {
    let args = Args::parse();
    Logger::init(args.verbosity);

    let mut dataset: Dataset<29, 1> = Dataset::from_csv("data/creditcard.csv".into())?;
    dataset.normalize();

    info!("Loaded {} sequences.", dataset.sequences.len());

    let mut model = RNN::<29, 10, 1>::new(
        ActivationFunction::Tanh,
        LossFunction::MSE,
        SGD::new(0.001).boxed(),
    );

    let start = std::time::Instant::now();
    model.train(
        &dataset,
        100,
        0.8,
        10,
        BinaryClassificationEvaluator { threshold: 0.5 },
    );
    let duration = start.elapsed();
    info!("Training completed in: {:?}", duration);

    // let mut dataset: Dataset<3, 1> = Dataset::from_csv("data/sample.csv".into())?;
    // dataset.normalize();

    // info!("Loaded {} sequences.", dataset.sequences.len());

    // let mut model = RNN::<3, 5, 1>::new(
    //     ActivationFunction::ReLU,
    //     LossFunction::MSE,
    //     SGD::new(0.000001).boxed(),
    // );

    // let start = std::time::Instant::now();
    // model.train(
    //     &dataset,
    //     1_000_000,
    //     0.8,
    //     50_000,
    //     BinaryClassificationEvaluator { threshold: 0.5 },
    // );
    // let duration = start.elapsed();
    // info!("Training completed in: {:?}", duration);

    // let mut dataset: Dataset<6, 1> = Dataset::from_csv("data/temp.csv".into())?;
    // dataset.normalize();

    // info!("Loaded {} sequences.", dataset.sequences.len());

    // let mut model = RNN::<6, 32, 1>::new(
    //     ActivationFunction::Sigmoid,
    //     LossFunction::MSE,
    //     SGD::new(0.0001).boxed(),
    // );

    // let start = std::time::Instant::now();
    // model.train(&dataset, 10000, 0.8, 1000, RegressionEvaluator);
    // let duration = start.elapsed();
    // info!("Training completed in: {:?}", duration);

    Ok(())
}
