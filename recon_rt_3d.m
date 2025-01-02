addpath ./mfile

ccc

if ~isfolder('./recon_data')
    mkdir recon_data
end

all_mat = dir('./raw_data/*3d_real_time*.mat');
nfile = length(all_mat);

%% recon parameters
FOV_recon = [220, 220] * 2;
nos_one = 12; % number of spirals per time frame
weight_ttv = 0.04;
weight_stv = 0.002;
para.setting.ifplot         = 0;
para.setting.ifGPU          = 1;


%% recon 
for ii = 1:nfile
    file_name = fullfile(all_mat(ii).folder, all_mat(ii).name);
    load(file_name)
    
    para.kspace_info = kspace_info;
    
    res = [kspace_info.user_ResolutionX, kspace_info.user_ResolutionY];
    sz = kspace_info.user_nKzEncoding;
    matrix_size = [round(FOV_recon ./ res / 2) * 2, sz];
    
    kspace = kspace * 1e3;
    
    %% coil compression
    sx  = size(kspace, 1);
    nos = size(kspace, 2);
    nc  = size(kspace, 3);
    n_virtual_coil = 6;
    
    kspace = reshape(kspace, [sx * nos, nc]);
    [~, ~, v] = svd(kspace, 'econ');
    kspace = kspace * v(:, 1:n_virtual_coil);
    kspace = reshape(kspace, [sx, nos, n_virtual_coil]);
    nc = size(kspace, 3);
    
    %%
    
    kx = kspace_info.kx_GIRF * matrix_size(1);
    ky = kspace_info.ky_GIRF * matrix_size(2);
    kz = kspace_info.RFIndex + 1;
    
    nframe = nos / nos_one;
    nframe = floor(nframe);
    
    nos         = nframe * nos_one;
    kspace      = kspace(:, 1:nos, :);
    view_order  = kspace_info.viewOrder(1:nos);
    kx          = kx(:, view_order);
    ky          = ky(:, view_order);
    
    %% demod delay
    q0 = kspace_info.user_QuaternionW;
    q1 = kspace_info.user_QuaternionX;
    q2 = kspace_info.user_QuaternionY;
    q3 = kspace_info.user_QuaternionZ;
    
    rot = [2 * (q0^2  + q1^2 ) - 1,     2 * (q1*q2 - q0*q3),        2 * (q1*q3 + q0*q2);
        2 * (q1*q2 + q0*q3),         2 * (q0^2  + q2^2 ) - 1,    2 * (q2*q3 - q0*q1);
        2 * (q1*q3 - q0*q2),         2 * (q2*q3 + q0*q1),        2 * (q0^2  + q3^2) - 1];
    
    dx = kspace_info.user_TranslationX;
    dy = kspace_info.user_TranslationY;
    dz = kspace_info.user_TranslationZ;
    
    kx0 = kspace_info.kx;
    ky0 = kspace_info.ky;
    
    kx0 = kx0 / kspace_info.user_ResolutionX;
    ky0 = ky0 / kspace_info.user_ResolutionY;
    
    kx_girf = kspace_info.kx_GIRF;
    ky_girf = kspace_info.ky_GIRF;
    
    kx_girf = kx_girf / kspace_info.user_ResolutionX;
    ky_girf = ky_girf / kspace_info.user_ResolutionY;
    
    % rotate trajectory to physical coordinate
    [nsample, narm] = size(kx0);
    k0 = cat(1, kx0(:)', ky0(:)', zeros(size(kx0(:)))');
    k0 = rot * k0;
    
    kx0 = reshape(k0(1, :), [nsample, narm]);
    ky0 = reshape(k0(2, :), [nsample, narm]);
    kz0 = reshape(k0(3, :), [nsample, narm]);
    
    k_girf = cat(1, kx_girf(:)', ky_girf(:)', zeros(size(kx_girf(:)))');
    k_girf = rot * k_girf;
    
    kx_girf = reshape(k_girf(1, :), [nsample, narm]);
    ky_girf = reshape(k_girf(2, :), [nsample, narm]);
    kz_girf = reshape(k_girf(3, :), [nsample, narm]);
    
    
    phase_x_0 = 2 * pi * dx * kx0;
    phase_y_0 = 2 * pi * dy * ky0;
    phase_z_0 = 2 * pi * dz * kz0;
    
    phase_x_girf = 2 * pi * dx * kx_girf;
    phase_y_girf = 2 * pi * dy * ky_girf;
    phase_z_girf = 2 * pi * dz * kz_girf;
    
    demod_phase_x = circshift(phase_x_0, [-2, 0]) - phase_x_girf;
    demod_phase_y = circshift(phase_y_0, [-2, 0]) - phase_y_girf;
    demod_phase_z = circshift(phase_z_0, [-2, 0]) - phase_z_girf;
    
    demod_phase = demod_phase_x + demod_phase_y + demod_phase_z;
    delay_corr = demod_phase(:, view_order);
    delay_corr = exp(-1i * delay_corr);
    kspace = kspace .* delay_corr;
    
    
    %% pre-allocate kSpace, kx, ky, w
    kspace_3d = zeros(sx, 1, sz, nframe, nc, 'single');
    kx_3d = zeros(sx, 1, sz, nframe);
    ky_3d = zeros(sx, 1, sz, nframe);
    
    % put the spirals at correcte location. This should also work for random
    % order.
    for i = 1:nos
        slice_idx = kz(i);
        frame_idx = ceil(i/nos_one);
        
        kSpace_temp = kspace(:,i,:);
        kx_temp = kx(:,i,:);
        ky_temp = ky(:,i,:);
        
        ns = sum(kspace_3d(1,:,slice_idx,frame_idx,1)~=0)+1;
        
        kspace_3d(:,ns,slice_idx,frame_idx,:) = kSpace_temp;
        kx_3d(:,ns,slice_idx,frame_idx,:) = kx_temp;
        ky_3d(:,ns,slice_idx,frame_idx,:) = ky_temp;
    end
    
    
    image = zeros([matrix_size, nframe, nc], 'single');
    for i = 1:sz
        for j = 1:size(kspace_3d, 2)
            kx_temp = permute(kx_3d(:,j,i,:),[1,2,4,3]);
            ky_temp = permute(ky_3d(:,j,i,:),[1,2,4,3]);
            kspace_temp = permute(kspace_3d(:,j,i,:,:),[1,2,4,5,3]);
            N = NUFFT.init(kx_temp, ky_temp, 1, [4,4], matrix_size(1), matrix_size(1));
            N.W = kspace_info.DCF(:, 1);
            N_all(j, i) = N;
            image_temp = NUFFT.NUFFT_adj(kspace_temp, N);
            image(:,:,i,:,:) = image(:,:,i,:,:) + permute(image_temp, [1, 2, 5, 3, 4]);
        end
    end
    
    
    image = fftshift(image, 3);
    image = ifft(image,[],3);
    image = fftshift(image,3);
    
    sens = get_sens_map(image, '3D');
    
    Data.kSpace = kspace_3d;
    Data.mask = Data.kSpace(1, :, :, :, 1) ~= 0;
    Data.N = N_all;
    Data.sens_map = sens;
    Data.first_est = image .* conj(sens);
    Data.first_est = sum(Data.first_est,5);
    
    scale = max(abs(Data.first_est(:)));
    
    % parameters
    
    para.Recon.weight_tTV       = scale * weight_ttv; % temporal regularization weight
    para.Recon.weight_sTV       = scale * weight_stv; % spatial regularization weight
    para.Recon.weight_sliceTV   = scale * weight_stv; % slice regularization weight
    para.Recon.epsilon          = eps('single');
    para.Recon.step_size        = 2;
    para.Recon.ifContinue       = 0;
    para.Recon.noi              = 150; % number of iterations
    para.Recon.type             = '3D SOS new'; %stack of spiral
    para.Recon.no_comp          = nc;
    para.Recon.break            = 1;
    
    [image_recon, para] = STCR_conjugate_gradient_3D(Data, para);
    
    image_recon = gather(image_recon);
    image_recon = abs(image_recon);
    image_recon = fliplr(rot90(image_recon, -1));
    save(sprintf('./recon_data/%s_narm_%g_ttv_%04g_stv_%04g_240_fov.mat', all_mat(ii).name(1:end-4), nos_one, weight_ttv, weight_stv), 'image_recon', 'para', '-v7.3')
    
    clear image N_all
    
    
    
end