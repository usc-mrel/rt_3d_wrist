function [Image,para] = STCR_conjugate_gradient_3D(Data,para)
%--------------------------------------------------------------------------
%   [Image,para] = STCR_conjugate_gradient_3D(Data,para)
%--------------------------------------------------------------------------
%   Solve MRI reconstruction problem using a conjugate gradient method.
%--------------------------------------------------------------------------
%   Inputs (for a 2D dynamic radial case):
%       - Data                      [structure] 
%           Data.kSpace             [sd, nor, nos, nof, nc]
%           Data.sens_map           [1, 1, 1, 1, nc]
%           Data.first_est          [sx, sy, sz, nof]
%           Data.N                  [NUFFT structure]
%
%               'sd'    number of readout points
%               'nor'   number of data per slice per time frame
%               'nos'   number of slices per time frame
%               'nof'   number of time frames
%               'nc'    number of coils
%               'sx'    image matrix size x
%               'sy'    image matrix size x
%               'sz'    image matrix size z
%           
%       - para                      [structure]
%           para.setting            [structure]
%               setting.ifplot      [0 or 1]
%               setting.ifGPU       [0 or 1]
%           para.Recon              [structure]
%               Recon.weight_tTV    [scalar]
%               Recon.weight_sTV    [scalar]
%               Recon.epsilon       [scalar]
%               Recon.step_size     [scalar]
%               Recon.noi           [positive integer]
%               Recon.type          [string]
%
%       - Data
%           Data.kSpace             measured k-space data "d"
%           Data.sens_map           sensitivity map
%           Data.first_est          initial estimation of "x": "A^H d"
%           Data.N                  NUFFT structure (see +NUFFT)
%
%       -para
%           para.setting.ifplot     display reconstruction process
%           para.setting.ifGPU      run function on a NVIDIA GPU
%           para.Recon.weight_tTV   "lambda_t"
%           para.Recon.weight_sTV   "lambda_s"
%           para.Recon.epsilon      "epsilon"
%           para.Recon.step_size    initial CG update step size
%           para.Recon.noi          number of iterations
%           para.Recon.type         reconstruction type see 
%                                   'compute_fidelity_ye_new'
%--------------------------------------------------------------------------
%   Output:
%       - Image     [sx, sy, sz, nof]
%       - para      [structure]
%
%       - Image     reconstructed images "m"
%--------------------------------------------------------------------------
%   A standard cost function it solves is the spatially and temporally
%   constrained reconstruction (STCR):
%
%   || Am - d ||_2^2 + lambda_t || TV_t m ||_1 + lambda_s || TV_s m ||_1
%
%   "A"         sampling matrix includes sensitivity maps, Fourier 
%               transform, and undersampling mask
%   "m"         image to be reconstructed
%   "d"         measured k-space data
%   ||.||_2^2   l2 norm
%   ||.||_1     l1 norm
%   "lambda_t"  temporal constraint weight
%   "lambda_s"  sparial constraint weight
%   TV_t        temporal total variation (TV) operator (finite difference)
%               sqrt( abs(m_t+1 - m_t)^2 + epsilon )
%   "epsilon"   small term to aviod singularity
%   TV_s        spatial TV operator
%               sqrt( abs(m_x+1 - m_x)^2 + abs(m_y+1 - m_y) + epsilon )
%--------------------------------------------------------------------------
%   Reference:
%       [1]     Acquisition and reconstruction of undersampled radial data 
%               for myocardial perfusion MRI. JMRI, 2009, 29(2):466-473.
%--------------------------------------------------------------------------
%   Author:
%       Ye Tian
%       E-mail: phye1988@gmail.com
%--------------------------------------------------------------------------

if isempty(Data.first_est)
    Image = [];
    return
end

fprintf([repmat('-', [1, 75]), '\n'])
disp('begin iterative STCR conjugate gradient reconstruction...');
fprintf([repmat('-', [1, 75]), '\n'])

disp_freq               = 10;
ifplot                  = para.setting.ifplot;
ifGPU                   = para.setting.ifGPU;
weight_tTV              = para.Recon.weight_tTV;
weight_sTV              = para.Recon.weight_sTV;
weight_slTV             = para.Recon.weight_sliceTV;
beta_sqrd               = para.Recon.epsilon;
para.Recon.step_size    = para.Recon.step_size(1);

new_img_x = single(Data.first_est);

if isfield(Data,'first_guess')
    new_img_x = Data.first_guess;
end

if isfield(Data,'phase_mod')
    Data.phase_mod_conj = conj(single(Data.phase_mod));
end
if isfield(Data,'sens_map')
    Data.sens_map_conj = conj(Data.sens_map);
end

if ifGPU
    new_img_x = gpuArray(new_img_x);

    if isfield(Data,'N')
        for i=1:size(Data.N, 1)
            for j = 1:size(Data.N, 2)
                Data.N(i, j).S = gpuArray(Data.N(i, j).S);
                Data.N(i, j).Apodizer = gpuArray(Data.N(i, j).Apodizer);
                Data.N(i, j).W = gpuArray(Data.N(i, j).W);
            end
        end
    end

end

para.Cost = struct('fidelityNorm',[],'temporalNorm',[],'spatialNorm',[],'totalCost',[]);
fprintf(' Iteration       Cost       Step    Time(s) \n')

for iter_no = 1:para.Recon.noi

    if mod(iter_no,disp_freq) == 1 || iter_no == 1 || disp_freq == 1
        t1 = tic;
    end
    
%% fidelity term/temporal/spatial TV
    tic;
    [update_term, fidelity_norm] = compute_fidelity_yt_new(new_img_x,Data,para);
    para.CPUtime.fidelity(iter_no) = toc;
    
    tic;
    update_term = update_term + compute_3DtTV_yt(new_img_x,weight_tTV,beta_sqrd);
    para.CPUtime.tTV(iter_no) = toc;
    
    tic;
    update_term = update_term + compute_sTV_yt(new_img_x,weight_sTV,beta_sqrd);
    update_term = update_term + compute_sliceTV_yt(new_img_x,weight_slTV,beta_sqrd);
    para.CPUtime.sTV(iter_no) = toc;

%% conjugate gradient
    tic;
    if iter_no > 1
        beta = update_term(:)'*update_term(:)/(update_term_old(:)'*update_term_old(:)+eps('single'));
        update_term = update_term + beta*update_term_old;
    end
    update_term_old = update_term; % clear update_term
    
%% line search   
    para.Cost = Cost_STCR_3D(fidelity_norm, new_img_x, weight_sTV, weight_tTV, weight_slTV, para.Cost); clear fidelity_update
    step_size = line_search(new_img_x, update_term_old, Data, para);
    para.Recon.step_size(iter_no) = step_size;

    new_img_x = new_img_x + step_size * update_term_old;
    para.CPUtime.update(iter_no) = toc;

%% plot & save 
    if ifplot ==1
        showImage3D(new_img_x,para.Cost)
    end
    
    % break when step size too small or cost not changing too much
    if iter_no > 1 && para.Recon.break
        if step_size<1e-4 %|| abs(para.Cost.totalCost(end) - para.Cost.totalCost(end-1))/para.Cost.totalCost(end-1) < 1e-4
            break
        end
    end
    
    if mod(iter_no,disp_freq) == 0 || iter_no == para.Recon.noi
        fprintf(sprintf('%10.0f %10.2f %10.4f %10.2f \n',iter_no,para.Cost.totalCost(end),step_size,toc(t1)));
    end
end

Image = squeeze(gather(new_img_x));
para = get_CPU_time(para);
fprintf(['Iterative STCR running time is ' num2str(para.CPUtime.interative_recon) 's' '\n'])
