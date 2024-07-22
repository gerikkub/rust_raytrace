

fn main() {
    cxx_build::bridge("src/main.rs")
        .cuda(true)
        .cudart("static")
        .flag("-gencode").flag("arch=compute_75,code=sm_75")
        .file("src/cuda_rt.cu")
        .flag_if_supported("-std c++11")
        .flag("--expt-relaxed-constexpr")
        .debug(true)
        .opt_level(0)
        .compile("cuda_rt");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_rt.cu");
    println!("cargo:rerun-if-changed=include/cuda_rt.h");

    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    println!(r"cargo:rustc-link-search=lib");
}