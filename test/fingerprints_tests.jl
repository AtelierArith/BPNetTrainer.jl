@testitem "FingerPrint Structure and Creation" begin
    using BPNetTrainer
    using BPNetTrainer: FingerPrint, FingerPrintParams
    
    # Test basic FingerPrint creation with Chebyshev basis
    atomtype = "Ti"
    envtypes = ["Ti", "O"]
    radial_Rc = 4.0
    radial_N = 10
    angular_Rc = 3.5
    angular_N = 8
    
    fp = FingerPrint(
        atomtype, envtypes;
        basistype="Chebyshev",
        radial_Rc=radial_Rc,
        radial_N=radial_N, 
        angular_Rc=angular_Rc,
        angular_N=angular_N
    )
    
    @test fp isa FingerPrint
    @test fp.atomtype == atomtype
    @test fp.envtypes == envtypes
    @test fp.nenv == length(envtypes)
    @test fp.itype == 1  # Ti is first in envtypes
    @test fp.sftype == "Chebyshev"
    @test fp.rc_max >= max(radial_Rc, angular_Rc)
    
    # Test calculated dimensions
    expected_coeffs = radial_N + angular_N + 2
    if length(envtypes) > 1
        expected_coeffs *= 2
    end
    @test fp.nsf == expected_coeffs
    @test length(fp.sfval_min) == expected_coeffs
    @test length(fp.sfval_max) == expected_coeffs
    @test length(fp.sfval_avg) == expected_coeffs
    @test length(fp.sfval_cov) == expected_coeffs
end

@testitem "FingerPrint Parameter Validation" begin
    using BPNetTrainer
    using BPNetTrainer: FingerPrint
    
    atomtype = "O"
    envtypes = ["Ti", "O"]
    
    # Test missing required parameters
    @test_throws KeyError FingerPrint(atomtype, envtypes; basistype="Chebyshev")
    @test_throws KeyError FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=4.0)
    @test_throws KeyError FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=4.0, radial_N=10)
    
    # Test invalid atom type
    @test_throws AssertionError FingerPrint("Zr", envtypes; 
        basistype="Chebyshev", radial_Rc=4.0, radial_N=10, angular_Rc=3.5, angular_N=8)
    
    # Test unsupported basis type
    @test_throws ErrorException FingerPrint(atomtype, envtypes; 
        basistype="InvalidBasis", radial_Rc=4.0, radial_N=10, angular_Rc=3.5, angular_N=8)
end

@testitem "Chebyshev Parameter Calculation" begin
    using BPNetTrainer
    using BPNetTrainer: chebyshevparam, NUM_MAX_FINGERPRINT_PARAMS, NENV_MAX
    
    nenv = 2
    kwargs = Dict(
        :radial_Rc => 4.0,
        :radial_N => 10,
        :angular_Rc => 3.5,
        :angular_N => 8
    )
    
    sfparam, sfenv, Rc, num_of_coeffs, num_of_parameters = chebyshevparam(nenv, kwargs)
    
    @test size(sfparam) == (NUM_MAX_FINGERPRINT_PARAMS, num_of_coeffs)
    @test size(sfenv) == (NENV_MAX, num_of_coeffs)
    @test Rc == max(kwargs[:angular_Rc], kwargs[:radial_Rc])
    @test num_of_coeffs == (kwargs[:radial_N] + kwargs[:angular_N] + 2) * 2  # *2 for nenv > 1
    @test num_of_parameters == NUM_MAX_FINGERPRINT_PARAMS
    
    # Check that parameters are correctly stored
    @test sfparam[1, 1] == kwargs[:radial_Rc]
    @test sfparam[2, 1] == kwargs[:radial_N]
    @test sfparam[3, 1] == kwargs[:angular_Rc]
    @test sfparam[4, 1] == kwargs[:angular_N]
end

@testitem "Single FingerPrint Info Generation" begin
    using BPNetTrainer
    using BPNetTrainer: FingerPrint, get_singlefingerprints_info, FingerPrintParams
    
    # Create a test fingerprint
    atomtype = "Ti"
    envtypes = ["Ti", "O"]
    fp = FingerPrint(
        atomtype, envtypes;
        basistype="Chebyshev",
        radial_Rc=4.0,
        radial_N=10,
        angular_Rc=3.5,
        angular_N=8
    )
    
    inputdim = fp.nsf
    fp_params = get_singlefingerprints_info(fp, inputdim)
    
    @test length(fp_params) == 1
    @test fp_params[1] isa FingerPrintParams
    @test fp_params[1].basistype == "any single basis"
    @test fp_params[1].num_kinds == 1
    @test fp_params[1].numparams == inputdim
    @test fp_params[1].startindex == 1
    @test fp_params[1].endindex == length(fp.sfparam[:, 1])
    @test length(fp_params[1].params) == size(fp.sfparam, 1)
end

@testitem "Multi FingerPrint Info Generation" begin
    using BPNetTrainer
    using BPNetTrainer: get_multifingerprints_info, FingerPrintParams, NUM_MAX_FINGERPRINT_PARAMS
    
    # Create a mock multi-fingerprint with proper structure
    # This simulates what would be loaded from a file
    mock_fingerprint = BPNetTrainer.FingerPrint(
        1, "test", "Ti", 2, ["Ti", "O"], 0.5, 8.0, "Multi", 
        20, 15,  # nsf, nsfparam
        zeros(Int64, 20),  # sf
        zeros(Float64, 15, 20),  # sfparam
        zeros(Int64, 2, 20),  # sfenv
        100,  # neval
        zeros(Float64, 20),  # sfval_min
        zeros(Float64, 20),  # sfval_max
        zeros(Float64, 20),  # sfval_avg
        zeros(Float64, 20)   # sfval_cov
    )
    
    # Set up multi-fingerprint parameters
    # First parameter should be 0.0 to indicate multi-version
    mock_fingerprint.sfparam[1, 1] = 0.0
    mock_fingerprint.sfparam[2, 1] = 2.0  # num_kinds = 2
    
    # First basis (Chebyshev)
    mock_fingerprint.sfparam[3, 1] = 1.0   # basis type = 1 (Chebyshev)
    mock_fingerprint.sfparam[4, 1] = 10.0  # numparams = 10
    mock_fingerprint.sfparam[5:8, 1] .= [4.0, 10.0, 3.5, 8.0]  # parameters
    
    # Second basis (Spline)
    start_idx = 3 + (NUM_MAX_FINGERPRINT_PARAMS + 2)
    mock_fingerprint.sfparam[start_idx, 1] = 2.0     # basis type = 2 (Spline)
    mock_fingerprint.sfparam[start_idx+1, 1] = 10.0  # numparams = 10
    mock_fingerprint.sfparam[start_idx+2:start_idx+5, 1] .= [3.0, 8.0, 2.5, 6.0]
    
    fp_params = get_multifingerprints_info(mock_fingerprint)
    
    @test length(fp_params) == 2
    
    # Test first basis
    @test fp_params[1].basistype == "Chebyshev"
    @test fp_params[1].num_kinds == 2
    @test fp_params[1].numparams == 10
    @test fp_params[1].startindex == 1
    @test fp_params[1].endindex == 10
    
    # Test second basis  
    @test fp_params[2].basistype == "Spline"
    @test fp_params[2].num_kinds == 2
    @test fp_params[2].numparams == 10
    @test fp_params[2].startindex == 11
    @test fp_params[2].endindex == 20
end

@testitem "FingerPrint Constants" begin
    using BPNetTrainer
    using BPNetTrainer: NUM_MAX_FINGERPRINT_PARAMS, NENV_MAX
    
    # Test that constants are properly defined
    @test NUM_MAX_FINGERPRINT_PARAMS isa Int
    @test NENV_MAX isa Int
    @test NUM_MAX_FINGERPRINT_PARAMS > 0
    @test NENV_MAX > 0
    
    # Test typical values (based on current implementation)
    @test NUM_MAX_FINGERPRINT_PARAMS == 4
    @test NENV_MAX == 2
end