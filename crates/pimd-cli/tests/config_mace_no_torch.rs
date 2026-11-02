//! `kind = "mace"` without the feature `torch` (MILESTONE_6 "Config": "only with feature `torch`;
//! otherwise a config error naming the feature"). Runs in the default build only.
#![cfg(not(feature = "torch"))]

mod mace_common;

use pimd_cli::Config;

/// P-6 config, build isolation: without the feature, a `kind = "mace"` config (otherwise valid)
/// is a config error whose message names the feature `torch`.
#[test]
fn mace_without_torch_feature_names_the_feature() {
    let toml = mace_common::mace_water_toml(&mace_common::tiny_export(), 10);
    match Config::from_toml_str(&toml) {
        Ok(c) => panic!("kind = \"mace\" accepted without the feature: {c:?}"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("torch"),
                "the error must name the feature `torch`: {msg:?}"
            );
        }
    }
}
