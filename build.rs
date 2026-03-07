#[cfg(feature = "cuda")]
fn build_cuda() {
    // Compile CUDA kernels via cc crate with CUDA support
    cc::Build::new()
        .cuda(true)
        .flag("-gencode=arch=compute_60,code=sm_60")  // Pascal (P100)
        .flag("-gencode=arch=compute_70,code=sm_70")  // Volta (V100)
        .flag("-gencode=arch=compute_80,code=sm_80")  // Ampere (A100)
        .flag("-gencode=arch=compute_89,code=sm_89")  // Ada (L40, RTX 4090)
        .flag("-gencode=arch=compute_90,code=sm_90")  // Hopper (H100)
        .flag("-Wno-deprecated-gpu-targets")
        .file("cuda/models.cu")
        .file("cuda/gp.cu")
        .compile("lightcurve_cuda");

    // Link against the CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=cuda/models.cu");
    println!("cargo:rerun-if-changed=cuda/gp.cu");
}

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
}
