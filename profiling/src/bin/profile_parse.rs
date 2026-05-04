//! Profiling harness: parse one fixture in a tight loop. Pick which one
//! via the FIXTURE env var (twitter, citm, or canada). Defaults to all.

use std::hint::black_box;

use bumpalo::Bump;
use datavalue_rs::DataValue;

const TWITTER: &str = include_str!("../../../benches/fixtures/twitter.json");
const CITM: &str = include_str!("../../../benches/fixtures/citm_catalog.json");
const CANADA: &str = include_str!("../../../benches/fixtures/canada.json");

fn parse_loop(name: &str, input: &str, iters: usize) {
    let mut arena = Bump::new();
    let start = std::time::Instant::now();
    for _ in 0..iters {
        arena.reset();
        let v = DataValue::from_str(black_box(input), &arena).unwrap();
        black_box(v);
    }
    let elapsed = start.elapsed();
    eprintln!(
        "{name}: {iters} iters in {elapsed:?} ({:.0?} per iter)",
        elapsed / iters as u32,
    );
}

fn main() {
    let fixture = std::env::var("FIXTURE").unwrap_or_default();
    match fixture.as_str() {
        "twitter" => parse_loop("twitter", TWITTER, 8000),
        "citm" => parse_loop("citm", CITM, 3000),
        "canada" => parse_loop("canada", CANADA, 1200),
        _ => {
            parse_loop("twitter", TWITTER, 4000);
            parse_loop("citm", CITM, 1500);
            parse_loop("canada", CANADA, 600);
        }
    }
}
