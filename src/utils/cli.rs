use clap::Parser;
use log::LevelFilter;

#[derive(Parser, Debug)]
#[command(name = "rnn-rust")]
pub struct Args {
    /// Sets the logger's verbosity level
    #[arg(short, long, value_name = "VERBOSITY", default_value_t = LevelFilter::Info)]
    pub verbosity: LevelFilter,
}
