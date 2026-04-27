//! twitter.json benchmark — DataValue vs serde_json::Value.
//!
//! Run with: `cargo bench --bench twitter`.

use bumpalo::Bump;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use datavalue::DataValue;

const TWITTER: &str = include_str!("twitter.json");

fn bench_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse twitter.json");
    group.throughput(Throughput::Bytes(TWITTER.len() as u64));

    group.bench_function("datavalue (fresh arena)", |b| {
        b.iter(|| {
            let arena = Bump::new();
            let v = DataValue::from_str(black_box(TWITTER), &arena).unwrap();
            black_box(v);
        });
    });

    group.bench_function("datavalue (reused arena)", |b| {
        let mut arena = Bump::new();
        b.iter(|| {
            arena.reset();
            let v = DataValue::from_str(black_box(TWITTER), &arena).unwrap();
            black_box(v);
        });
    });

    group.bench_function("serde_json::Value", |b| {
        b.iter(|| {
            let v: serde_json::Value = serde_json::from_str(black_box(TWITTER)).unwrap();
            black_box(v);
        });
    });

    group.finish();
}

fn bench_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("access twitter.json");

    let arena = Bump::new();
    let dv = DataValue::from_str(TWITTER, &arena).unwrap();
    let sj: serde_json::Value = serde_json::from_str(TWITTER).unwrap();

    group.bench_function("datavalue walk statuses", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(items) = dv.get("statuses").and_then(|v| v.as_array()) {
                for s in items {
                    if let Some(name) = s
                        .get("user")
                        .and_then(|u| u.get("screen_name"))
                        .and_then(|v| v.as_str())
                    {
                        acc = acc.wrapping_add(name.len() as u64);
                    }
                    if let Some(rc) = s.get("retweet_count").and_then(|v| v.as_i64()) {
                        acc = acc.wrapping_add(rc as u64);
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("serde_json walk statuses", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(items) = sj.get("statuses").and_then(|v| v.as_array()) {
                for s in items {
                    if let Some(name) = s
                        .get("user")
                        .and_then(|u| u.get("screen_name"))
                        .and_then(|v| v.as_str())
                    {
                        acc = acc.wrapping_add(name.len() as u64);
                    }
                    if let Some(rc) = s.get("retweet_count").and_then(|v| v.as_i64()) {
                        acc = acc.wrapping_add(rc as u64);
                    }
                }
            }
            black_box(acc);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_parse, bench_access);
criterion_main!(benches);
