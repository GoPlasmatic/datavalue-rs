//! Profiling harness: convert a pre-built `serde_json::Value` tree into
//! an `OwnedDataValue` in a tight loop. Pick which fixture via the
//! FIXTURE env var (twitter, citm, or canada). Defaults to all. Also
//! pick MODE=borrow (default) or MODE=move.

use std::hint::black_box;

use datavalue_rs::OwnedDataValue;

const TWITTER: &str = include_str!("../../../benches/fixtures/twitter.json");
const CITM: &str = include_str!("../../../benches/fixtures/citm_catalog.json");
const CANADA: &str = include_str!("../../../benches/fixtures/canada.json");

fn convert_loop_borrow(name: &str, input: &str, iters: usize) {
    let sj: serde_json::Value = serde_json::from_str(input).unwrap();
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let v = OwnedDataValue::from(black_box(&sj));
        black_box(v);
    }
    let elapsed = start.elapsed();
    eprintln!(
        "{name} borrow: {iters} iters in {elapsed:?} ({:.0?} per iter)",
        elapsed / iters as u32,
    );
}

fn convert_loop_move(name: &str, input: &str, iters: usize) {
    let sj: serde_json::Value = serde_json::from_str(input).unwrap();
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let cloned = sj.clone();
        let v = OwnedDataValue::from(black_box(cloned));
        black_box(v);
    }
    let elapsed = start.elapsed();
    eprintln!(
        "{name} move: {iters} iters in {elapsed:?} ({:.0?} per iter, incl. clone)",
        elapsed / iters as u32,
    );
}

fn main() {
    let fixture = std::env::var("FIXTURE").unwrap_or_default();
    let mode = std::env::var("MODE").unwrap_or_else(|_| "borrow".to_string());
    let run = |name: &str, input: &str, iters: usize| match mode.as_str() {
        "move" => convert_loop_move(name, input, iters),
        _ => convert_loop_borrow(name, input, iters),
    };
    match fixture.as_str() {
        "twitter" => run("twitter", TWITTER, 4000),
        "citm" => run("citm", CITM, 2000),
        "canada" => run("canada", CANADA, 800),
        _ => {
            run("twitter", TWITTER, 2000);
            run("citm", CITM, 1000);
            run("canada", CANADA, 400);
        }
    }
}
