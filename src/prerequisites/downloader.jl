"""
    download_dataset()

Download and extract the example TiO₂ dataset for BPNet training.

This function downloads a compressed example dataset from the AENET project
containing TiO₂ structures with Chebyshev basis functions. The dataset is
automatically decompressed and extracted for immediate use.

# Process
1. Downloads `aenet-example-02-TiO2-Chebyshev.tar.bz2` from ann.atomistic.net
2. Decompresses the bzip2 archive
3. Extracts the tar file to `extracted_files/` directory
4. Skips download if files already exist

# Files Created
- `aenet-example-02-TiO2-Chebyshev.tar.bz2`: Original compressed archive
- `aenet-example-02-TiO2-Chebyshev.tar`: Decompressed tar file
- `extracted_files/`: Directory containing extracted dataset files

# Storage Location
Files are saved to the dataset root directory (`DATASET_ROOT[]`), typically:
- `~/.julia/scratchspaces/<uuid>/dataset/` on Unix systems

# Example
```julia
using BPNetTrainer
download_dataset()  # Downloads and extracts example TiO₂ data
```

# See also
- [`generate_example_dataset()`](@ref): Process the downloaded data for training
- [`switchdatasetdir!()`](@ref): Change the download directory
"""
function download_dataset()
    url = "http://ann.atomistic.net/files/aenet-example-02-TiO2-Chebyshev.tar.bz2"
    bz2name = "aenet-example-02-TiO2-Chebyshev.tar.bz2"
    tarname = first(splitext(bz2name))
    bz2path = joinpath(DATASET_ROOT[], bz2name)
    tarpath = joinpath(DATASET_ROOT[], tarname)
    extractpath = joinpath(DATASET_ROOT[], "extracted_files")
    if !isfile(bz2path)
        Downloads.download(url, bz2path)
        open(bz2path) do inp
            open(tarpath, "w") do output_file
                stream = TranscodingStreams.TranscodingStream(
                    CodecBzip2.Bzip2Decompressor(),
                    inp,
                )
                write(output_file, stream)
            end
        end
        # Extract to the new directory
        Tar.extract(tarpath, extractpath)
        @info "Done!"
    else
        @info "Dataset exists in $(bz2path)"
    end
end
