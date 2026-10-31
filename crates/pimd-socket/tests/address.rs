//! Address parsing (MILESTONE_5 "Required tests: Address parsing"; SPEC K-1): `"unix:/path"` is a
//! Unix-domain socket, `"host:port"` a TCP socket kept as written; malformed strings are errors
//! (`Address::Err = PotentialError`).

use std::path::PathBuf;

use pimd_socket::Address;

/// K-1: `unix:/path` → `Address::Unix(path)`, the path after the prefix taken verbatim.
#[test]
fn unix_addresses() {
    for (s, p) in [
        ("unix:/tmp/pimd-water.sock", "/tmp/pimd-water.sock"),
        ("unix:/a/b c/d.sock", "/a/b c/d.sock"),
        ("unix:/x", "/x"),
    ] {
        assert_eq!(
            s.parse::<Address>()
                .unwrap_or_else(|e| panic!("{s}: {e:?}")),
            Address::Unix(PathBuf::from(p)),
            "{s}"
        );
    }
}

/// K-1: `host:port` → `Address::Tcp` with the string as written (host name or IPv4 literal).
#[test]
fn tcp_addresses() {
    for s in [
        "localhost:31415",
        "127.0.0.1:31415",
        "example.org:1",
        "localhost:65535",
    ] {
        assert_eq!(
            s.parse::<Address>()
                .unwrap_or_else(|e| panic!("{s}: {e:?}")),
            Address::Tcp(s.to_string()),
            "{s}"
        );
    }
}

/// K-1: malformed strings are errors: empty, `unix:` without a path, no port, empty port, empty
/// host, a port that is not a number, out of range (65536) or negative.
#[test]
fn malformed_addresses_rejected() {
    for s in [
        "",
        "unix:",
        "localhost",
        "localhost:",
        ":31415",
        "localhost:port",
        "localhost:65536",
        "localhost:-1",
        "localhost:31415x",
    ] {
        assert!(s.parse::<Address>().is_err(), "{s:?} accepted");
    }
}
