//! Config for `kind = "mace"`, cases added after the M6b review (MILESTONE_6 "After review", N4;
//! Q24: the model file is read and hashed when the config is parsed). Feature `torch`.
#![cfg(feature = "torch")]

mod mace_common;

use mace_common::{mace_water_toml, tiny_export};
use pimd_cli::{CliError, Config};

/// N4: a `potential.model` that does not exist is `CliError::Io` at parse time, and the message
/// names the key and the path.
#[test]
fn missing_model_file_is_io_error_at_parse() {
    let missing = tiny_export().with_file_name("no_such_export.pt");
    match Config::from_toml_str(&mace_water_toml(&missing, 10)) {
        Err(CliError::Io(e)) => {
            let m = e.to_string();
            assert!(
                m.contains("potential.model"),
                "message {m:?} must name potential.model"
            );
            assert!(
                m.contains("no_such_export.pt"),
                "message {m:?} must name the path"
            );
        }
        other => panic!("expected CliError::Io, got {other:?}"),
    }
}
