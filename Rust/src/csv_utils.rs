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
            "approach",
            "n",
            "m",
            "avg_time",
            "std_time",
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

pub fn calculate_stats_without_outliers(values: &[f64], z_threshold: f64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    // Calculate initial statistics with all data points
    let initial_stats = calculate_stats(values);
    let initial_mean = initial_stats.0;
    let initial_std_dev = initial_stats.1;

    // If we only have one value or std_dev is zero, return the initial stats
    if values.len() == 1 || initial_std_dev == 0.0 {
        return initial_stats;
    }

    // Filter out the outliers
    let filtered_values: Vec<f64> = values
        .iter()
        .filter(|&x| {
            let z_score = (x - initial_mean).abs() / initial_std_dev;
            z_score <= z_threshold
        })
        .cloned()
        .collect();

    // If all values were considered outliers (unlikely but possible),
    // return the original statistics
    println!("After filtering, length={}",filtered_values.len());
    if filtered_values.is_empty() {
        return initial_stats;
    }

    // Calculate statistics on the filtered dataset
    calculate_stats(&filtered_values)
}