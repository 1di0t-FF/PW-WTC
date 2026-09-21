function run_pwwtc_batch(inputMat, outputMat)
%RUN_PWWTC_BATCH  PW-WTC batch entry for the comparison archive.
%  Adds this folder (holding compute_matlab_wcoherence_batch.m, the exact
%  kernel that produced the published gammas) and delegates.  The window
%  matrices and all options come from run_comparison.py.
addpath(fileparts(mfilename('fullpath')));
compute_matlab_wcoherence_batch(inputMat, outputMat);
end
