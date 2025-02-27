use std::fs::OpenOptions;
use std::path::Path;
use csv::{Writer, ReaderBuilder};


pub fn write_data(
    filename: &str,
    approach: &str,
    n: usize,
    m: usize,
    avg_time: f64,
    std_time: f64,
) {
    let file_exists = Path::new(filename).exists();

    let mut writer = Writer::from_writer(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)
            .unwrap()
    );

    if !file_exists {
        writer.write_record(&[
            "approach\t",
            "n\t",
            "m\t",
            "avg_time\t",
            "std_time\t",
        ]).unwrap();
    }

    writer.write_record(&[
        approach.to_string(),
        n.to_string(),
        m.to_string(),
        avg_time.to_string(),
       std_time.to_string(),
    ]).unwrap();

    writer.flush().unwrap();
}

pub fn calculate_stats(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let avg = values.iter().sum::<f64>() / values.len() as f64;

    if values.len() == 1 {
        return (avg, 0.0);
    }

    let variance = values.iter()
        .map(|x| (x - avg).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;

    let std_dev = variance.sqrt();

    (avg, std_dev)
}