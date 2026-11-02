//! Config for `kind = "mace"` (MILESTONE_6 "Config"; SPEC P-6, D-3 torch note). Feature `torch`.
//!
//! Keys: `model` (required), `device` "cpu" | "cuda" | "cuda:N" (default "cpu"), `dtype`
//! "float64" | "float32" (default "float64"), `threads` ≥ 1 (default 1), `deterministic` (default
//! true); `[system] symbols` required. Validation (errors name the field as in Q15): unknown
//! symbol, `threads = 0`, bad device string, mace keys with another kind, missing `symbols`.
//!
//! Error naming as in `config_m5.rs`: out-of-range values are `InvalidValue { what:
//! "section.key" }` where the brief's Q15 rule makes that unambiguous (`threads = 0`); for string
//! values (device, dtype, symbols) and misplaced keys the brief does not say whether the error is
//! `InvalidValue` or `Parse`, so `rejected_naming` accepts an `InvalidValue` naming the field or a
//! `Parse` whose message mentions its key (assumption, as in M5).
#![cfg(feature = "torch")]

mod common;
mod mace_common;

use std::path::PathBuf;

use common::{BASE, parse, remove, set, water_toml};
use mace_common::{mace_water_toml, tiny_export};
use pimd_cli::{CliError, Config, PotentialConfig};
use pimd_torch::{Device, Dtype};

fn err(toml: &str) -> CliError {
    match Config::from_toml_str(toml) {
        Ok(c) => panic!("config accepted: {c:?}\n{toml}"),
        Err(e) => e,
    }
}

/// Rejected with `InvalidValue` naming one of `fields`, or `Parse` mentioning one of their keys.
fn rejected_naming(toml: &str, fields: &[&str]) {
    match err(toml) {
        CliError::InvalidValue { what, .. } => {
            assert!(
                fields.contains(&what.as_str()),
                "named {what}, want one of {fields:?}"
            )
        }
        CliError::Parse(m) => assert!(
            fields
                .iter()
                .any(|f| m.contains(f.rsplit('.').next().unwrap())),
            "Parse message {m:?} names none of {fields:?}"
        ),
        other => panic!("expected InvalidValue/Parse naming {fields:?}, got {other:?}"),
    }
}

fn base() -> String {
    mace_water_toml(&tiny_export(), 10)
}

/// (model, device, dtype, threads, deterministic) of a mace config.
fn mace_fields(c: &Config) -> (PathBuf, Device, Dtype, usize, bool) {
    match &c.potential {
        PotentialConfig::Mace {
            model,
            device,
            dtype,
            threads,
            deterministic,
            ..
        } => (model.clone(), *device, *dtype, *threads, *deterministic),
        other => panic!("expected kind = \"mace\", got {other:?}"),
    }
}

/// P-6 config (brief example): every key given; `model` kept as the given path; symbols kept.
#[test]
fn mace_example_parses() {
    let t = set(&base(), "potential", "device", "\"cpu\"");
    let t = set(&t, "potential", "dtype", "\"float64\"");
    let t = set(&t, "potential", "threads", "1");
    let t = set(&t, "potential", "deterministic", "true");
    let c = parse(&t);
    assert_eq!(
        mace_fields(&c),
        (tiny_export(), Device::Cpu, Dtype::F64, 1, true)
    );
    assert_eq!(
        c.system.symbols,
        Some(vec!["O".to_string(), "H".to_string(), "H".to_string()])
    );
    assert!(c.system.cell.is_none());
}

/// P-6 config defaults: device "cpu", dtype "float64", threads 1, deterministic true.
#[test]
fn mace_defaults() {
    let c = parse(&base());
    assert_eq!(
        mace_fields(&c),
        (tiny_export(), Device::Cpu, Dtype::F64, 1, true)
    );
}

/// P-6 config: "cuda" is CUDA device 0, "cuda:N" device N; "float32"; threads 4; deterministic
/// false.
#[test]
fn mace_values() {
    for (s, d) in [
        ("cpu", Device::Cpu),
        ("cuda", Device::Cuda(0)),
        ("cuda:0", Device::Cuda(0)),
        ("cuda:1", Device::Cuda(1)),
        ("cuda:12", Device::Cuda(12)),
    ] {
        let c = parse(&set(&base(), "potential", "device", &format!("\"{s}\"")));
        assert_eq!(mace_fields(&c).1, d, "device {s:?}");
    }
    for (s, d) in [("float32", Dtype::F32), ("float64", Dtype::F64)] {
        let c = parse(&set(&base(), "potential", "dtype", &format!("\"{s}\"")));
        assert_eq!(mace_fields(&c).2, d, "dtype {s:?}");
    }
    let c = parse(&set(&base(), "potential", "threads", "4"));
    assert_eq!(mace_fields(&c).3, 4);
    let c = parse(&set(&base(), "potential", "deterministic", "false"));
    assert!(!mace_fields(&c).4);
}

/// P-6 config validation: bad device strings are rejected, naming `potential.device`.
#[test]
fn bad_device_rejected() {
    parse(&base()); // the unmodified mace config is accepted (else the rejections are vacuous)
    for v in [
        "gpu", "CUDA", "Cpu", "cuda:", "cuda:x", "cuda:-1", "cuda:1.5", " cpu", "", "cuda0",
        "cpu:0",
    ] {
        rejected_naming(
            &set(&base(), "potential", "device", &format!("\"{v}\"")),
            &["potential.device"],
        );
    }
}

/// P-6 config validation: dtypes other than "float64" and "float32" are rejected, naming
/// `potential.dtype`.
#[test]
fn bad_dtype_rejected() {
    parse(&base()); // the unmodified mace config is accepted (else the rejections are vacuous)
    for v in ["float16", "f64", "double", "Float32", ""] {
        rejected_naming(
            &set(&base(), "potential", "dtype", &format!("\"{v}\"")),
            &["potential.dtype"],
        );
    }
}

/// P-6 config validation (Q15): `threads = 0` is `InvalidValue { what: "potential.threads" }`; a
/// negative or non-integer value is a type error (`Parse`).
#[test]
fn threads_validated() {
    match err(&set(&base(), "potential", "threads", "0")) {
        CliError::InvalidValue { what, .. } => assert_eq!(what, "potential.threads"),
        other => panic!("expected InvalidValue(potential.threads), got {other:?}"),
    }
    for v in ["-1", "1.5", "\"two\""] {
        assert!(
            matches!(
                err(&set(&base(), "potential", "threads", v)),
                CliError::Parse(_)
            ),
            "threads = {v}"
        );
    }
}

/// P-6 config validation: `symbols` are required with kind = "mace" and must be element symbols
/// (H … I, `pimd_torch::atomic_number`), else rejected naming `system.symbols`. A known element
/// that the model lacks parses (the model is checked when it is loaded; `run_mace.rs`).
#[test]
fn symbols_required_and_known() {
    parse(&base()); // the unmodified mace config is accepted (else the rejections are vacuous)
    rejected_naming(&remove(&base(), "symbols"), &["system.symbols"]);
    for v in [
        r#"["O", "H", "Xx"]"#,
        r#"["O", "H", "Xe"]"#,
        r#"["O", "H", "h"]"#,
        r#"["O", "H", "D"]"#,
        r#"["O1", "H", "H"]"#,
    ] {
        rejected_naming(&set(&base(), "system", "symbols", v), &["system.symbols"]);
    }
    parse(&set(&base(), "system", "symbols", r#"["O", "H", "F"]"#));
}

/// P-6 config: `model` is required with kind = "mace" (`Parse` naming it).
#[test]
fn model_required() {
    parse(&base()); // the unmodified mace config is accepted (else the rejections are vacuous)
    rejected_naming(&remove(&base(), "model"), &["potential.model"]);
}

/// P-6 config validation: `model`, `device`, `dtype`, `threads` and `deterministic` belong to
/// kind = "mace": with kind = "lj" (M4 neon) or "socket" (M5 water) each is rejected, naming it.
#[test]
fn mace_keys_need_mace_kind() {
    parse(&base()); // the unmodified mace config is accepted (else the rejections are vacuous)
    let model = format!("\"{}\"", tiny_export().display());
    let keys: [(&str, &str); 5] = [
        ("model", &model),
        ("device", "\"cpu\""),
        ("dtype", "\"float64\""),
        ("threads", "1"),
        ("deterministic", "true"),
    ];
    for other in [BASE.to_string(), water_toml("unix:/tmp/pimd-m6.sock", 2)] {
        for (k, v) in keys {
            rejected_naming(
                &set(&other, "potential", k, v),
                &[&format!("potential.{k}")],
            );
        }
    }
}

/// P-6 config validation: LJ and socket keys with kind = "mace" are rejected (`Parse` or
/// `InvalidValue` naming the key).
#[test]
fn other_kinds_keys_rejected_with_mace() {
    parse(&base()); // the unmodified mace config is accepted (else the rejections are vacuous)
    for (k, v) in [
        ("epsilon_kelvin", "35.6"),
        ("sigma_angstrom", "2.749"),
        ("cutoff_sigma", "2.0"),
        ("skin_sigma", "0.25"),
        ("shift", "true"),
        ("address", "\"unix:/tmp/x.sock\""),
        ("n_clients", "2"),
        ("timeout_s", "10.0"),
    ] {
        rejected_naming(
            &set(&base(), "potential", k, v),
            &[&format!("potential.{k}")],
        );
    }
}
