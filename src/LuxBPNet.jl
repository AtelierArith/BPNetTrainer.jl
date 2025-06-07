module LuxBPNet

using Downloads
using Tar: Tar
using CodecBzip2: CodecBzip2
using TranscodingStreams: TranscodingStreams
using Scratch: @get_scratch!

const DATASET_DIR = Ref{String}()

function download_dataset()
	url = "http://ann.atomistic.net/files/aenet-example-02-TiO2-Chebyshev.tar.bz2"
	bz2name = "aenet-example-02-TiO2-Chebyshev.tar.bz2"
	tarname = first(splitext(bz2name))
	bz2path = joinpath(DATASET_DIR[], bz2name)
	tarpath = joinpath(DATASET_DIR[], tarname)
	extractpath = joinpath(DATASET_DIR[], "extracted_files")
	if !isfile(bz2path)
		Downloads.download(url, bz2path)
		open(bz2path) do inp
			open(tarpath, "w") do output_file
	            stream = TranscodingStreams.TranscodingStream(CodecBzip2.Bzip2Decompressor(), inp)
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

function __init__()
	global DATASET_DIR
    DATASET_DIR[] = @get_scratch!("dataset")
end

end # module LuxBPNet
