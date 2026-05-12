//! Profiling harness: convert a pre-built OwnedDataValue tree into an
//! arena-bound DataValue in a tight loop. Pick which fixture via the
//! FIXTURE env var (twitter, citm, or canada). Defaults to all.

use std::hint::black_box;

use bumpalo::Bump;
use datavalue_rs::OwnedDataValue;

const TWITTER: &str = include_str!("../../../benches/fixtures/twitter.json");
const CITM: &str = include_str!("../../../benches/fixtures/citm_catalog.json");
const CANADA: &str = include_str!("../../../benches/fixtures/canada.json");

fn convert_loop(name: &str, input: &str, iters: usize) {
    let owned = OwnedDataValue::from_json(input).unwrap();
    let mut arena = Bump::new();
    let start = std::time::Instant::now();
    for _ in 0..iters {
        arena.reset();
        let v = black_box(&owned).to_arena(&arena);
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
        "twitter" => convert_loop("twitter", TWITTER, 8000),
        "citm" => convert_loop("citm", CITM, 4000),
        "canada" => convert_loop("canada", CANADA, 1500),
        _ => {
            convert_loop("twitter", TWITTER, 4000);
            convert_loop("citm", CITM, 2000);
            convert_loop("canada", CANADA, 800);
        }
    }
}
