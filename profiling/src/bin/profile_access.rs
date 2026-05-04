//! Profiling harness: parse each fixture once, then walk a representative
//! path many times. Per-walk functions are `#[inline(never)]` so the
//! profile shows where time goes inside each shape.

use std::hint::black_box;

use bumpalo::Bump;
use datavalue_rs::DataValue;

const TWITTER: &str = include_str!("../../../benches/fixtures/twitter.json");
const CITM: &str = include_str!("../../../benches/fixtures/citm_catalog.json");
const CANADA: &str = include_str!("../../../benches/fixtures/canada.json");

#[inline(never)]
fn walk_twitter(v: &DataValue<'_>) -> u64 {
    let mut s = 0u64;
    if let Some(items) = v["statuses"].as_array() {
        for it in items {
            if let Some(name) = it["user"]["screen_name"].as_str() {
                s = s.wrapping_add(name.len() as u64);
            }
            if let Some(rc) = it["retweet_count"].as_i64() {
                s = s.wrapping_add(rc as u64);
            }
        }
    }
    s
}

#[inline(never)]
fn walk_citm(v: &DataValue<'_>) -> u64 {
    let mut s = 0u64;
    if let Some(events) = v["events"].as_object() {
        for (_, ev) in events {
            if let Some(id) = ev["id"].as_i64() {
                s = s.wrapping_add(id as u64);
            }
            if let Some(arr) = ev["subTopicIds"].as_array() {
                s = s.wrapping_add(arr.len() as u64);
            }
        }
    }
    s
}

#[inline(never)]
fn walk_canada(v: &DataValue<'_>) -> u64 {
    let mut s = 0u64;
    if let Some(features) = v["features"].as_array() {
        for f in features {
            if let Some(coords) = f["geometry"]["coordinates"].as_array() {
                s = s.wrapping_add(coords.len() as u64);
            }
        }
    }
    s
}

fn time_walk<F: Fn() -> u64>(name: &str, iters: usize, f: F) {
    let start = std::time::Instant::now();
    let mut acc: u64 = 0;
    for _ in 0..iters {
        acc = acc.wrapping_add(black_box(f()));
    }
    black_box(acc);
    let elapsed = start.elapsed();
    eprintln!(
        "{name}: {iters} iters in {elapsed:?} ({:.0?} per iter)",
        elapsed / iters as u32,
    );
}

fn main() {
    let arena = Bump::new();
    let twitter = DataValue::from_str(TWITTER, &arena).unwrap();
    let citm = DataValue::from_str(CITM, &arena).unwrap();
    let canada = DataValue::from_str(CANADA, &arena).unwrap();

    time_walk("twitter", 400_000, || walk_twitter(black_box(&twitter)));
    time_walk("citm", 1_500_000, || walk_citm(black_box(&citm)));
    time_walk("canada", 8_000_000, || walk_canada(black_box(&canada)));
}
