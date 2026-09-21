function compute_matlab_wcoherence_batch(inputMat, outputMat)
%COMPUTE_MATLAB_WCOHERENCE_BATCH  PW-WTC gamma for a batch of MP windows.
%  Input .mat:  x_windows [N x 600] Base-day MP, y_windows Test-day MP,
%               Fs, freq_limits, voices_per_oct, wavelet_type, power_weight_alpha.
%  Per window: mirror-pad to 1200 samples (kills the edge effects; only the
%  original 600 columns are used), take wcoherence(yp, xp) and cwt(xp), then
%  gamma = sum(rho .* (|W_base|^2)^a) / sum((|W_base|^2)^a) over every
%  (scale,time) point inside the cone of influence.  a = 0.5 published.
%  Output .mat: gamma [N x 1].  Serial on purpose: the parallel pool is
%  unstable here and serial keeps the published numbers bit-reproducible.

S = load(inputMat);
x_windows = double(S.x_windows);
y_windows = double(S.y_windows);
Fs = double(S.Fs);
freq_limits = double(S.freq_limits);
voices_per_oct = double(S.voices_per_oct);
wavelet_type = char(S.wavelet_type);
power_weight_alpha = double(S.power_weight_alpha);
if ~isscalar(power_weight_alpha) || ~isfinite(power_weight_alpha) || power_weight_alpha < 0
    error('power_weight_alpha must be a finite non-negative scalar.');
end
if size(x_windows, 1) ~= size(y_windows, 1) || size(x_windows, 2) ~= size(y_windows, 2)
    error('x_windows and y_windows must have the same size.');
end

n_windows = size(x_windows, 1);
window_size = size(x_windows, 2);
gamma = zeros(n_windows, 1);
for i = 1:n_windows
    xp = x_windows(i, :).';
    yp = y_windows(i, :).';
    xp = [xp; flipud(xp)];
    yp = [yp; flipud(yp)];
    [wcoh, wcs, freqs, coi] = wcoherence(yp, xp, Fs, ...
        'FrequencyLimits', freq_limits, 'VoicesPerOctave', voices_per_oct);
    wtx = cwt(xp, Fs, wavelet_type, ...
        'FrequencyLimits', freq_limits, 'VoicesPerOctave', voices_per_oct);
    rho = double(wcoh(:, 1:window_size));
    phi = double(angle(wcs(:, 1:window_size)));
    amp = double(abs(wtx(:, 1:window_size)));
    coi_part = double(coi(1:window_size));
    valid = double(freqs(:)) >= coi_part(:).';
    rho_vec = rho(valid);
    phi_vec = phi(valid);
    amp_vec = amp(valid);
    finite = isfinite(rho_vec) & isfinite(phi_vec) & isfinite(amp_vec);
    rho_vec = rho_vec(finite);
    amp_vec = amp_vec(finite);
    if isempty(rho_vec)
        gamma(i) = NaN;
        continue;
    end
    power_weights = (amp_vec .^ 2) .^ power_weight_alpha;
    gamma(i) = sum(rho_vec .* power_weights, 'omitnan') / ...
        (sum(power_weights, 'omitnan') + eps);
end

save(outputMat, 'gamma', '-v7');
end
