fn main() {
    // Invalidate the build script when the thrift file changes
    println!("cargo:rerun-if-changed=thrift/user.thrift");

    // Use the system `thrift` compiler to generate Rust code into OUT_DIR
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by Cargo");
    let status = std::process::Command::new("thrift")
        .args(["--gen", "rs", "-out", &out_dir, "thrift/user.thrift"])
        .status()
        .expect("failed to run thrift compiler. Is `thrift` installed?");
    if !status.success() {
        panic!("thrift compiler failed: {status}");
    }

    // Post-process generated file to convert inner crate attributes (#![...]) to outer ones (#[])
    // so the generated code can be safely included in a module.
    let gen_file = std::path::Path::new(&out_dir).join("user.rs");
    if let Ok(src) = std::fs::read_to_string(&gen_file) {
        let patched = src.replace("#![", "#[");
        let _ = std::fs::write(&gen_file, patched);
    }
}
