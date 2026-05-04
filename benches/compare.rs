//! Cross-library JSON benchmarks.
//!
//! Compares datavalue-rs against serde_json, simd-json, sonic-rs, and
//! json-rust across four workloads — parse, serialize, access, mutate —
//! over the canonical fixtures (twitter, citm_catalog, canada).
//!
//! Run with: `cargo bench --bench compare --features serde_json`.

use bumpalo::Bump;
use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use datavalue_rs::{DataValue, OwnedDataValue};
// simd_json's prelude is intentionally not imported at module scope: it
// provides traits (ValueAccess, ValueAsContainer, ...) that also impl on
// sonic_rs::Value via the shared `value-trait` crate, which causes method
// ambiguity. Each simd_json bench body imports the prelude locally.
use sonic_rs::{JsonContainerTrait, JsonValueTrait};

const TWITTER: &str = include_str!("fixtures/twitter.json");
const CITM: &str = include_str!("fixtures/citm_catalog.json");
const CANADA: &str = include_str!("fixtures/canada.json");

const FIXTURES: &[(&str, &str)] = &[("twitter", TWITTER), ("citm", CITM), ("canada", CANADA)];

// -----------------------------------------------------------------------------
// 1. Deserialize: JSON string -> in-memory DOM.
// -----------------------------------------------------------------------------

fn bench_deserialize(c: &mut Criterion) {
    for (name, input) in FIXTURES {
        let mut group = c.benchmark_group(format!("parse/{name}"));
        group.throughput(Throughput::Bytes(input.len() as u64));

        group.bench_function("datavalue (fresh arena)", |b| {
            b.iter(|| {
                let arena = Bump::new();
                let v = DataValue::from_str(black_box(input), &arena).unwrap();
                black_box(v);
            });
        });

        group.bench_function("datavalue (reused arena)", |b| {
            let mut arena = Bump::new();
            b.iter(|| {
                arena.reset();
                let v = DataValue::from_str(black_box(input), &arena).unwrap();
                black_box(v);
            });
        });

        group.bench_function("serde_json::Value", |b| {
            b.iter(|| {
                let v: serde_json::Value = serde_json::from_str(black_box(input)).unwrap();
                black_box(v);
            });
        });

        // simd-json mutates its input buffer, so each iteration needs a fresh copy.
        group.bench_function("simd_json (borrowed)", |b| {
            b.iter_batched(
                || input.as_bytes().to_vec(),
                |mut buf| {
                    let v = simd_json::to_borrowed_value(&mut buf).unwrap();
                    black_box(v);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function("simd_json (owned)", |b| {
            b.iter_batched(
                || input.as_bytes().to_vec(),
                |mut buf| {
                    let v = simd_json::to_owned_value(&mut buf).unwrap();
                    black_box(v);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function("sonic_rs::Value", |b| {
            b.iter(|| {
                let v: sonic_rs::Value = sonic_rs::from_str(black_box(input)).unwrap();
                black_box(v);
            });
        });

        group.bench_function("json-rust", |b| {
            b.iter(|| {
                let v = json::parse(black_box(input)).unwrap();
                black_box(v);
            });
        });

        group.finish();
    }
}

// -----------------------------------------------------------------------------
// 2. Serialize: in-memory DOM -> JSON string.
// -----------------------------------------------------------------------------

fn bench_serialize(c: &mut Criterion) {
    for (name, input) in FIXTURES {
        let mut group = c.benchmark_group(format!("serialize/{name}"));

        // Pre-parse once per fixture; the bench only measures emit cost.
        let arena = Bump::new();
        let dv = DataValue::from_str(input, &arena).unwrap();
        let sj: serde_json::Value = serde_json::from_str(input).unwrap();
        let sn: sonic_rs::Value = sonic_rs::from_str(input).unwrap();
        let jr = json::parse(input).unwrap();
        let mut sb = input.as_bytes().to_vec();
        let smv = simd_json::to_owned_value(&mut sb).unwrap();

        // Use serde_json's compact output as the canonical size for throughput.
        let canon = serde_json::to_string(&sj).unwrap();
        group.throughput(Throughput::Bytes(canon.len() as u64));

        group.bench_function("datavalue", |b| {
            b.iter(|| {
                let s = dv.to_json_string();
                black_box(s);
            });
        });

        group.bench_function("serde_json::Value", |b| {
            b.iter(|| {
                let s = serde_json::to_string(&sj).unwrap();
                black_box(s);
            });
        });

        group.bench_function("simd_json (owned)", |b| {
            b.iter(|| {
                let s = simd_json::to_string(&smv).unwrap();
                black_box(s);
            });
        });

        group.bench_function("sonic_rs::Value", |b| {
            b.iter(|| {
                let s = sonic_rs::to_string(&sn).unwrap();
                black_box(s);
            });
        });

        group.bench_function("json-rust", |b| {
            b.iter(|| {
                let s = jr.dump();
                black_box(s);
            });
        });

        group.finish();
    }
}

// -----------------------------------------------------------------------------
// 3. Access: walk a representative path and accumulate.
// -----------------------------------------------------------------------------
//
// Each fixture has its own walk pattern shaped by the data:
//   twitter — sum screen_name lengths + retweet_count over statuses[]
//   citm    — iterate events object, sum id + subTopicIds.len() per event
//   canada  — sum vertex counts across features[].geometry.coordinates[]

fn bench_access(c: &mut Criterion) {
    bench_access_twitter(c);
    bench_access_citm(c);
    bench_access_canada(c);
}

fn bench_access_twitter(c: &mut Criterion) {
    let mut group = c.benchmark_group("access/twitter");
    let arena = Bump::new();
    let dv = DataValue::from_str(TWITTER, &arena).unwrap();
    let sj: serde_json::Value = serde_json::from_str(TWITTER).unwrap();
    let sn: sonic_rs::Value = sonic_rs::from_str(TWITTER).unwrap();
    let jr = json::parse(TWITTER).unwrap();
    let mut sb = TWITTER.as_bytes().to_vec();
    let smv = simd_json::to_owned_value(&mut sb).unwrap();

    group.bench_function("datavalue", |b| {
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

    group.bench_function("serde_json::Value", |b| {
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

    group.bench_function("simd_json (owned)", |b| {
        use simd_json::prelude::*;
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(items) = smv.get("statuses").as_array() {
                for s in items {
                    if let Some(name) = s.get("user").get("screen_name").as_str() {
                        acc = acc.wrapping_add(name.len() as u64);
                    }
                    if let Some(rc) = s.get("retweet_count").as_i64() {
                        acc = acc.wrapping_add(rc as u64);
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("sonic_rs::Value", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(items) = sn.get("statuses").as_array() {
                for s in items {
                    if let Some(name) = s.get("user").get("screen_name").as_str() {
                        acc = acc.wrapping_add(name.len() as u64);
                    }
                    if let Some(rc) = s.get("retweet_count").as_i64() {
                        acc = acc.wrapping_add(rc as u64);
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("json-rust", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            for s in jr["statuses"].members() {
                if let Some(name) = s["user"]["screen_name"].as_str() {
                    acc = acc.wrapping_add(name.len() as u64);
                }
                if let Some(rc) = s["retweet_count"].as_i64() {
                    acc = acc.wrapping_add(rc as u64);
                }
            }
            black_box(acc);
        });
    });

    group.finish();
}

fn bench_access_citm(c: &mut Criterion) {
    let mut group = c.benchmark_group("access/citm");
    let arena = Bump::new();
    let dv = DataValue::from_str(CITM, &arena).unwrap();
    let sj: serde_json::Value = serde_json::from_str(CITM).unwrap();
    let sn: sonic_rs::Value = sonic_rs::from_str(CITM).unwrap();
    let jr = json::parse(CITM).unwrap();
    let mut sb = CITM.as_bytes().to_vec();
    let smv = simd_json::to_owned_value(&mut sb).unwrap();

    group.bench_function("datavalue", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(events) = dv.get("events").and_then(|v| v.as_object()) {
                for (_, event) in events {
                    if let Some(id) = event.get("id").and_then(|v| v.as_i64()) {
                        acc = acc.wrapping_add(id as u64);
                    }
                    if let Some(st) = event.get("subTopicIds").and_then(|v| v.as_array()) {
                        acc = acc.wrapping_add(st.len() as u64);
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("serde_json::Value", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(events) = sj.get("events").and_then(|v| v.as_object()) {
                for (_, event) in events {
                    if let Some(id) = event.get("id").and_then(|v| v.as_i64()) {
                        acc = acc.wrapping_add(id as u64);
                    }
                    if let Some(st) = event.get("subTopicIds").and_then(|v| v.as_array()) {
                        acc = acc.wrapping_add(st.len() as u64);
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("simd_json (owned)", |b| {
        use simd_json::prelude::*;
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(events) = smv.get("events").as_object() {
                for (_, event) in events.iter() {
                    if let Some(id) = event.get("id").as_i64() {
                        acc = acc.wrapping_add(id as u64);
                    }
                    if let Some(st) = event.get("subTopicIds").as_array() {
                        acc = acc.wrapping_add(st.len() as u64);
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("sonic_rs::Value", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(events) = sn.get("events").as_object() {
                for (_, event) in events.iter() {
                    if let Some(id) = event.get("id").as_i64() {
                        acc = acc.wrapping_add(id as u64);
                    }
                    if let Some(st) = event.get("subTopicIds").as_array() {
                        acc = acc.wrapping_add(st.len() as u64);
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("json-rust", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            for (_, event) in jr["events"].entries() {
                if let Some(id) = event["id"].as_i64() {
                    acc = acc.wrapping_add(id as u64);
                }
                let st = &event["subTopicIds"];
                if st.is_array() {
                    acc = acc.wrapping_add(st.len() as u64);
                }
            }
            black_box(acc);
        });
    });

    group.finish();
}

fn bench_access_canada(c: &mut Criterion) {
    let mut group = c.benchmark_group("access/canada");
    let arena = Bump::new();
    let dv = DataValue::from_str(CANADA, &arena).unwrap();
    let sj: serde_json::Value = serde_json::from_str(CANADA).unwrap();
    let sn: sonic_rs::Value = sonic_rs::from_str(CANADA).unwrap();
    let jr = json::parse(CANADA).unwrap();
    let mut sb = CANADA.as_bytes().to_vec();
    let smv = simd_json::to_owned_value(&mut sb).unwrap();

    group.bench_function("datavalue", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(features) = dv.get("features").and_then(|v| v.as_array()) {
                for f in features {
                    if let Some(rings) = f
                        .get("geometry")
                        .and_then(|g| g.get("coordinates"))
                        .and_then(|c| c.as_array())
                    {
                        for ring in rings {
                            if let Some(pts) = ring.as_array() {
                                acc = acc.wrapping_add(pts.len() as u64);
                            }
                        }
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("serde_json::Value", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(features) = sj.get("features").and_then(|v| v.as_array()) {
                for f in features {
                    if let Some(rings) = f
                        .get("geometry")
                        .and_then(|g| g.get("coordinates"))
                        .and_then(|c| c.as_array())
                    {
                        for ring in rings {
                            if let Some(pts) = ring.as_array() {
                                acc = acc.wrapping_add(pts.len() as u64);
                            }
                        }
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("simd_json (owned)", |b| {
        use simd_json::prelude::*;
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(features) = smv.get("features").as_array() {
                for f in features {
                    if let Some(rings) = f.get("geometry").get("coordinates").as_array() {
                        for ring in rings {
                            if let Some(pts) = ring.as_array() {
                                acc = acc.wrapping_add(pts.len() as u64);
                            }
                        }
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("sonic_rs::Value", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            if let Some(features) = sn.get("features").as_array() {
                for f in features {
                    if let Some(rings) = f.get("geometry").get("coordinates").as_array() {
                        for ring in rings {
                            if let Some(pts) = ring.as_array() {
                                acc = acc.wrapping_add(pts.len() as u64);
                            }
                        }
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("json-rust", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            for f in jr["features"].members() {
                let rings = &f["geometry"]["coordinates"];
                if rings.is_array() {
                    for ring in rings.members() {
                        if ring.is_array() {
                            acc = acc.wrapping_add(ring.len() as u64);
                        }
                    }
                }
            }
            black_box(acc);
        });
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// 4. Mutate: increment retweet_count for every status in twitter.json.
// -----------------------------------------------------------------------------
//
// Only twitter is exercised here — the patch is a single, well-defined edit
// and that's enough to compare mutation costs across heap-DOM shapes.
//
// Arena `DataValue` is intentionally NOT in this group: it's immutable by
// design (the arena is the unit of mutation per CLAUDE.md), so any "mutate"
// benchmark for it would measure rebuild-into-fresh-arena, which is a
// different workflow from the in-place mutation the others exercise. Use
// `OwnedDataValue` when mutation is the actual requirement.
//
// sonic_rs is omitted: its mutation surface differs enough to make a fair
// comparable patch awkward. Add it in a follow-up if needed.

fn bench_mutate(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutate/twitter");
    group.throughput(Throughput::Bytes(TWITTER.len() as u64));

    let odv = OwnedDataValue::from_json(TWITTER).unwrap();
    group.bench_function("OwnedDataValue (clone + mutate)", |b| {
        b.iter_batched(
            || odv.clone(),
            |mut v| {
                bump_retweet_owned(&mut v);
                black_box(v);
            },
            BatchSize::SmallInput,
        );
    });

    let sj: serde_json::Value = serde_json::from_str(TWITTER).unwrap();
    group.bench_function("serde_json (clone + mutate)", |b| {
        b.iter_batched(
            || sj.clone(),
            |mut v| {
                if let Some(items) = v.get_mut("statuses").and_then(|s| s.as_array_mut()) {
                    for s in items {
                        if let Some(rc) = s.get_mut("retweet_count").and_then(|v| v.as_i64()) {
                            s["retweet_count"] = serde_json::Value::from(rc + 1);
                        }
                    }
                }
                black_box(v);
            },
            BatchSize::SmallInput,
        );
    });

    let mut sb = TWITTER.as_bytes().to_vec();
    let smv = simd_json::to_owned_value(&mut sb).unwrap();
    group.bench_function("simd_json (clone + mutate)", |b| {
        use simd_json::prelude::*;
        b.iter_batched(
            || smv.clone(),
            |mut v| {
                if let Some(items) = v.get_mut("statuses").and_then(|s| s.as_array_mut()) {
                    for s in items {
                        if let Some(rc) = s.get("retweet_count").as_i64()
                            && let Some(o) = s.as_object_mut()
                        {
                            o.insert("retweet_count".into(), simd_json::OwnedValue::from(rc + 1));
                        }
                    }
                }
                black_box(v);
            },
            BatchSize::SmallInput,
        );
    });

    let jr = json::parse(TWITTER).unwrap();
    group.bench_function("json-rust (clone + mutate)", |b| {
        b.iter_batched(
            || jr.clone(),
            |mut v| {
                let n = v["statuses"].len();
                for i in 0..n {
                    if let Some(rc) = v["statuses"][i]["retweet_count"].as_i64() {
                        v["statuses"][i]["retweet_count"] = (rc + 1).into();
                    }
                }
                black_box(v);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ---- OwnedDataValue in-place mutator ----

fn bump_retweet_owned(top: &mut OwnedDataValue) {
    let OwnedDataValue::Object(top_pairs) = top else {
        return;
    };
    for (k, v) in top_pairs.iter_mut() {
        if k == "statuses" {
            let OwnedDataValue::Array(items) = v else {
                continue;
            };
            for status in items.iter_mut() {
                let OwnedDataValue::Object(pairs) = status else {
                    continue;
                };
                for (sk, sv) in pairs.iter_mut() {
                    if sk == "retweet_count"
                        && let OwnedDataValue::Number(n) = sv
                        && let Some(i) = n.as_i64()
                    {
                        *sv = OwnedDataValue::Number(datavalue_rs::NumberValue::from_i64(i + 1));
                    }
                }
            }
        }
    }
}

criterion_group!(
    benches,
    bench_deserialize,
    bench_serialize,
    bench_access,
    bench_mutate
);
criterion_main!(benches);
