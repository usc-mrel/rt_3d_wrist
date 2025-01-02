function [fidelity_update,fidelity_norm] = compute_fidelity_yt_new(image,Data,para)
%--------------------------------------------------------------------------
%   [fidelity_update, fidelity_norm] 
%       = compute_fidelity_yt_new(image, Data, para)
%--------------------------------------------------------------------------
%   Compute fidelity update term of a MRI reconstruction problem
%--------------------------------------------------------------------------
%   Inputs:      
%       - image             [sx, sy, nof, ...]
%       - Data              [structure]
%       - para              [structure]
%           Recon.type      [string]
%
%       - image             image
%       - Data              see 'help STCR_conjugate_gradient.m'
%       - para              see 'help STCR_conjugate_gradient.m'
%       - para.Recon.type   reconstruction type
%--------------------------------------------------------------------------
%   Output:
%       - fidelity_update   [sx, sy, nof, ...]
%       - fidelity_norm     [scalar]
%
%       - fidelity_update   A^H (Am - d)
%       - fidelity_norm     || Am - d ||_2^2
%--------------------------------------------------------------------------
%   A standard fidelity term it solves is:
%
%   || Am - d ||_2^2
%
%   and the output is:
%
%   A^H (Am - d)
%
%   see 'help STCR_conjugate_gradient.m' for more information.
%--------------------------------------------------------------------------
%   Reference:
%       [1]     Acquisition and reconstruction of undersampled radial data 
%               for myocardial perfusion MRI. JMRI, 2009, 29(2):466-473.
%--------------------------------------------------------------------------
%   Author:
%       Ye Tian
%       E-mail: phye1988@gmail.com
%--------------------------------------------------------------------------

switch para.Recon.type
    case {'2D','2D less memory'}

        %if isfield(Data,'Apodizer')
        %    image = image.*Data.Apodizer;
        %end
        fidelity_update = bsxfun(@times,image,Data.sens_map);
        fidelity_update = fft2(fidelity_update,para.Recon.kSpace_size(1),para.Recon.kSpace_size(2));
        if isfield(Data,'phase_mod')
            fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod);
            fidelity_update = sum(fidelity_update,5);
            fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod_conj);
        else
            fidelity_update = bsxfun(@times, fidelity_update, Data.mask);
        end

        fidelity_update = Data.kSpace - fidelity_update;
        
%         fidelity_norm = fidelity_update/para.Recon.kSpace_size(1)/8;
        fidelity_norm = sqrt(sum(abs(fidelity_update(:)/para.Recon.kSpace_size(1)).^2));
                
        if isfield(Data,'filter')
            fidelity_update = fidelity_update .* Data.filter; % use filter to accelerate converge
        end
        
        fidelity_update = ifft2(fidelity_update);
        fidelity_update(para.Recon.image_size(1)+1:end,:,:,:,:) = [];
        fidelity_update(:,para.Recon.image_size(2)+1:end,:,:,:) = [];

        fidelity_update = bsxfun(@times, fidelity_update, Data.sens_map_conj);
        fidelity_update = sum(fidelity_update, 4);
        
        if isfield(Data,'Apodizer')
            fidelity_update = fidelity_update.*Data.Apodizer;
        end

    case 'seperate SMS'

        %fidelity_update = bsxfun(@times,image,Data.sens_map);
        fidelity_update = image.*Data.sens_map;
        fidelity_update = fft2(fidelity_update,para.Recon.kSpace_size(1),para.Recon.kSpace_size(2));
        
        fidelity_update_temp(:,:,:,:,1,1,1) = sum(fidelity_update,5);
        fidelity_update_temp(:,:,:,:,1,1,2) = fidelity_update(:,:,:,:,1,:,:) + exp(1i*2*pi/3)*fidelity_update(:,:,:,:,2,:,:) + exp(1i*4*pi/3)*fidelity_update(:,:,:,:,3,:,:);
        fidelity_update_temp(:,:,:,:,1,1,3) = fidelity_update(:,:,:,:,1,:,:) + exp(1i*4*pi/3)*fidelity_update(:,:,:,:,2,:,:) + exp(1i*2*pi/3)*fidelity_update(:,:,:,:,3,:,:);
        %fidelity_update_temp = bsxfun(@times,fidelity_update_temp,Data.mask);
        fidelity_update_temp = fidelity_update_temp.*Data.mask;
        
        fidelity_update_temp = Data.kSpace - fidelity_update_temp;
        % norm for line search
        fidelity_norm = sum(abs(fidelity_update_temp(:)).^2/prod(para.Recon.kSpace_size)/64);
        fidelity_norm = sqrt(fidelity_norm);
        % end
        fidelity_update(:,:,:,:,1) = sum(fidelity_update_temp,7);
        fidelity_update(:,:,:,:,2) = fidelity_update_temp(:,:,:,:,:,:,1) + exp(-1i*2*pi/3)*fidelity_update_temp(:,:,:,:,:,:,2) + exp(-1i*4*pi/3)*fidelity_update_temp(:,:,:,:,:,:,3);
        fidelity_update(:,:,:,:,3) = fidelity_update_temp(:,:,:,:,:,:,1) + exp(-1i*4*pi/3)*fidelity_update_temp(:,:,:,:,:,:,2) + exp(-1i*2*pi/3)*fidelity_update_temp(:,:,:,:,:,:,3);
        clear fidelity_update_temp
        if isfield(Data,'filter')
            %fidelity_update = bsxfun(@times,fidelity_update,Data.filter);
            fidelity_update = fidelity_update.*Data.filter;
        end
        fidelity_update = ifft2(fidelity_update);
        %fidelity_update = sum(bsxfun(@times,fidelity_update,Data.sens_map_conj),4);
        fidelity_update = sum(fidelity_update.*Data.sens_map_conj,4);
        if isfield(Data,'Apodizer')
            fidelity_update = fidelity_update.*Data.Apodizer;
        end
        
    case 'seperate SMS less memory'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;

        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:,:));

            %fidelity_update = fft2(fidelity_update,para.Recon.kSpace_size(1),para.Recon.kSpace_size(2));
            fidelity_update(para.Recon.image_size(1)+1:para.Recon.kSpace_size(1),para.Recon.image_size(2)+1:para.Recon.kSpace_size(2),:,:,:,:,:,:,:) = 0;
            fidelity_update = circshift(fidelity_update,[(para.Recon.kSpace_size(1)-para.Recon.image_size(1))/2,(para.Recon.kSpace_size(2)-para.Recon.image_size(2))/2]);
            fidelity_update = fft2(fidelity_update);
            
            fidelity_update_temp(:,:,:,1,1,:,1) = sum(fidelity_update,5);
            fidelity_update_temp(:,:,:,1,1,:,2) = fidelity_update(:,:,:,:,1,:,:) + exp(1i*2*pi/3)*fidelity_update(:,:,:,:,2,:,:) + exp(1i*4*pi/3)*fidelity_update(:,:,:,:,3,:,:);
            fidelity_update_temp(:,:,:,1,1,:,3) = fidelity_update(:,:,:,:,1,:,:) + exp(1i*4*pi/3)*fidelity_update(:,:,:,:,2,:,:) + exp(1i*2*pi/3)*fidelity_update(:,:,:,:,3,:,:);

            fidelity_update_temp = bsxfun(@times,fidelity_update_temp,Data.mask);
            fidelity_update_temp = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update_temp;
            
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update_temp(:)).^2/prod(para.Recon.kSpace_size)/64);
        
            fidelity_update(:,:,:,:,1,:) = sum(fidelity_update_temp,7);
            fidelity_update(:,:,:,:,2,:) = fidelity_update_temp(:,:,:,:,:,:,1) + exp(-1i*2*pi/3)*fidelity_update_temp(:,:,:,:,:,:,2) + exp(-1i*4*pi/3)*fidelity_update_temp(:,:,:,:,:,:,3);
            fidelity_update(:,:,:,:,3,:) = fidelity_update_temp(:,:,:,:,:,:,1) + exp(-1i*4*pi/3)*fidelity_update_temp(:,:,:,:,:,:,2) + exp(-1i*2*pi/3)*fidelity_update_temp(:,:,:,:,:,:,3);
            clear fidelity_update_temp
            
            if isfield(Data,'filter')
                fidelity_update = bsxfun(@times,fidelity_update,Data.filter);
            end
        
            fidelity_update = ifft2(fidelity_update);
            fidelity_update = circshift(fidelity_update,-[(para.Recon.kSpace_size(1)-para.Recon.image_size(1))/2,(para.Recon.kSpace_size(2)-para.Recon.image_size(2))/2]);
            
            fidelity_update(para.Recon.image_size(1)+1:end,:,:,:,:,:) = [];
            fidelity_update(:,para.Recon.image_size(2)+1:end,:,:,:,:) = [];
            fidelity_update_all = fidelity_update_all + bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i,:,:));
        end
        fidelity_update = fidelity_update_all;
        fidelity_norm = sqrt(fidelity_norm);
        return
        
    case 'Toeplitz SMS'
        fidelity_update_all = zeros(size(image),class(image));
        %image = image.*Data.Apodizer;
        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
            fidelity_update(para.Recon.image_size(1)+1:para.Recon.kSpace_size(1),para.Recon.image_size(2)+1:para.Recon.kSpace_size(2),:,:,:,:,:) = 0;

            fidelity_update = fft2(fidelity_update);
            
            fidelity_update_temp(:,:,:,:,1,1,1) = sum(fidelity_update,5);
            fidelity_update_temp(:,:,:,:,1,1,2) = fidelity_update(:,:,:,:,1,:,:) + exp(1i*2*pi/3)*fidelity_update(:,:,:,:,2,:,:) + exp(1i*4*pi/3)*fidelity_update(:,:,:,:,3,:,:);
            fidelity_update_temp(:,:,:,:,1,1,3) = fidelity_update(:,:,:,:,1,:,:) + exp(1i*4*pi/3)*fidelity_update(:,:,:,:,2,:,:) + exp(1i*2*pi/3)*fidelity_update(:,:,:,:,3,:,:);
 
            fidelity_update_temp = bsxfun(@times,fidelity_update_temp,Data.mask);
            fidelity_update_temp = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update_temp;
        
            fidelity_update(:,:,:,:,1) = sum(fidelity_update_temp,7);
            fidelity_update(:,:,:,:,2) = fidelity_update_temp(:,:,:,:,:,:,1) + exp(-1i*2*pi/3)*fidelity_update_temp(:,:,:,:,:,:,2) + exp(-1i*4*pi/3)*fidelity_update_temp(:,:,:,:,:,:,3);
            fidelity_update(:,:,:,:,3) = fidelity_update_temp(:,:,:,:,:,:,1) + exp(-1i*4*pi/3)*fidelity_update_temp(:,:,:,:,:,:,2) + exp(-1i*2*pi/3)*fidelity_update_temp(:,:,:,:,:,:,3);
            clear fidelity_update_temp

            fidelity_update = ifft2(fidelity_update);
            
            fidelity_update(para.Recon.image_size(1)+1:end,:,:,:,:) = [];
            fidelity_update(:,para.Recon.image_size(2)+1:end,:,:,:) = [];
            fidelity_update_all = fidelity_update_all + bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i,:));
        end
        fidelity_update_all = fidelity_update_all.*Data.Apodizer;
        fidelity_update = fidelity_update_all;

        return
        
    case 'seperate SMS projections less memory'
        fidelity_update_all = zeros(size(image),class(image));

        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));

            %fidelity_update = fft2(fidelity_update,para.Recon.kSpace_size(1),para.Recon.kSpace_size(2));
            fidelity_update(para.Recon.image_size(1)+1:para.Recon.kSpace_size(1),para.Recon.image_size(2)+1:para.Recon.kSpace_size(2),:,:,:,:,:,:,:) = 0;
            fidelity_update = circshift(fidelity_update,[(para.Recon.kSpace_size(1)-para.Recon.image_size(1))/2,(para.Recon.kSpace_size(2)-para.Recon.image_size(2))/2]);
            fidelity_update = fft2(fidelity_update);

            fidelity_update = fidelity_update.*Data.phase_mod;
            fidelity_update = sum(fidelity_update,5);

            fidelity_update = bsxfun(@times,fidelity_update,Data.mask);
            fidelity_update = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update;
        
            fidelity_update = fidelity_update.*Data.phase_mod_conj;
            
            if isfield(Data,'filter')
                fidelity_update = bsxfun(@times,fidelity_update,Data.filter);
            end
        
            fidelity_update = ifft2(fidelity_update);
            fidelity_update = circshift(fidelity_update,-[(para.Recon.kSpace_size(1)-para.Recon.image_size(1))/2,(para.Recon.kSpace_size(2)-para.Recon.image_size(2))/2]);
            
            fidelity_update(para.Recon.image_size(1)+1:end,:,:,:,:) = [];
            fidelity_update(:,para.Recon.image_size(2)+1:end,:,:,:) = [];
            fidelity_update_all = fidelity_update_all + bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i,:));
        end
        fidelity_update = fidelity_update_all;
        return
        
    case 'NUFFT less memory'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
            fidelity_update = NUFFT.NUFFT_new(fidelity_update,Data.N);
            if para.Recon.nSMS ~= 1
                fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod);
                fidelity_update = sum(fidelity_update,5);
                fidelity_update = Data.kSpace(:,:,:,i) - fidelity_update;
                fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)).^2)/288/288/64;
                fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod_conj);
            else
                fidelity_update = Data.kSpace(:,:,:,i) - fidelity_update;
                fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)).^2)/288/288/64;
            end
            fidelity_update = NUFFT.NUFFT_adj_new(fidelity_update,Data.N);
            fidelity_update_all = fidelity_update_all + bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i,:));
        end
        fidelity_norm = sqrt(fidelity_norm);
        fidelity_update = fidelity_update_all;
        return

    case 'NUFFT'
        fidelity_update = bsxfun(@times,image,Data.sens_map);
        fidelity_update = NUFFT.NUFFT(fidelity_update,Data.N);
        if para.Recon.nSMS ~= 1
            fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod);
            fidelity_update = sum(fidelity_update,5);
            fidelity_update = Data.kSpace - fidelity_update;
            fidelity_norm = sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
            fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod_conj);
        else
            fidelity_update = (Data.kSpace - fidelity_update).*logical(abs(Data.kSpace));%.*Data.kwic;
            fidelity_norm = sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
        end
        fidelity_norm = sqrt(fidelity_norm);
        fidelity_update = NUFFT.NUFFT_adj(fidelity_update,Data.N);
        if isfield(para.Recon, 'ssg')
            if para.Recon.ssg.flag
                fidelity_update = section_kernel_apply(fidelity_update, Data.ssg);
            end
        end
        fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj);
        fidelity_update = sum(fidelity_update,4);
        
    case 'NUFFT RSG'
        fidelity_update = bsxfun(@times,image,Data.sens_map);
        fidelity_update = NUFFT.NUFFT(fidelity_update,Data.N);

        fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod);
        fidelity_update = sum(fidelity_update,5);
        fidelity_update = Data.kSpace - fidelity_update;
        fidelity_norm = sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
        fidelity_norm = sqrt(fidelity_norm);
        
        fidelity_update = apply_radial_ssg_trajectroy_speficied(gather(fidelity_update), Data.phase_mod_idx, Data.theta, Data.rssg);
        if para.setting.ifGPU == 1
            fidelity_update = gpuArray(fidelity_update);
        end
        fidelity_update = NUFFT.NUFFT_adj(fidelity_update,Data.N);
        fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj);
        fidelity_update = sum(fidelity_update,4);
        
    case 'NUFFT coil'
%         fidelity_update = bsxfun(@times,image,Data.sens_map);
        fidelity_update = NUFFT.NUFFT(image,Data.N);
        if para.Recon.nSMS ~= 1
            fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod);
            fidelity_update = sum(fidelity_update,5);
            fidelity_update = Data.kSpace - fidelity_update;
            fidelity_norm = sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
            fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod_conj);
        else
            fidelity_update = (Data.kSpace - fidelity_update).*logical(abs(Data.kSpace));%.*Data.kwic;
            fidelity_norm = sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
        end
        fidelity_norm = sqrt(fidelity_norm);
        fidelity_update = NUFFT.NUFFT_adj(fidelity_update,Data.N);
%         fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj);
%         fidelity_update = sum(fidelity_update,4);
        
    case 'GNUFFT'
        fidelity_update = bsxfun(@times,image,Data.sens_map);
        fidelity_update = GROG.GNUFFT(fidelity_update,Data.G{1});
        fidelity_update = GROG.GNUFFT_adj(fidelity_update,Data.G{1});
        fidelity_update = sum(bsxfun(@times,fidelity_update,Data.sens_map_conj),4);
    case 'coil by coil'
        fidelity_update = fft2(image);
        fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod);
        fidelity_update = ifft2(fidelity_update);
    case '3D less memory'

        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
            fidelity_update = fft3(fidelity_update);
            fidelity_update = bsxfun(@times,fidelity_update,Data.mask);
            fidelity_update = Data.kSpace(:,:,:,:,i) - fidelity_update;
            fidelity_norm   = fidelity_norm + sum(abs(fidelity_update(:)).^2)/288/288/64;
            fidelity_update = fidelity_update.*Data.filter;
            fidelity_update = ifft3(fidelity_update);
            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,:,i));
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        fidelity_update = fidelity_update_all;
    case '3D'

        fidelity_update = bsxfun(@times,image,Data.sens_map);
        fidelity_update = fft3(fidelity_update);
        fidelity_update = bsxfun(@times,fidelity_update,Data.mask);
        fidelity_update = Data.kSpace - fidelity_update;
        fidelity_norm   = sum(abs(fidelity_update(:)).^2)/288/288/64;
        fidelity_norm   = sqrt(fidelity_norm)/2;
        fidelity_update = fidelity_update.*Data.filter;
        fidelity_update = ifft3(fidelity_update);
        fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj);
        fidelity_update = sum(fidelity_update,5);

        
    case 'MB5'
        
        fidelity_update_all = zeros(size(image),class(image));
        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
            fidelity_update = fft2(fidelity_update);
            fidelity_update = sum(fidelity_update.*Data.phase,5);
            fidelity_update = bsxfun(@times,fidelity_update,Data.mask);
            fidelity_update = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update;
            fidelity_update = sum(fidelity_update.*conj(Data.phase),7);
            clear fidelity_update_temp
            fidelity_update = bsxfun(@times,fidelity_update,Data.filter);
            fidelity_update = ifft2(fidelity_update);
            fidelity_update_all = fidelity_update_all + bsxfun(@times,fidelity_update,conj(Data.sens_map(:,:,:,i,:)));
        end
        fidelity_update = fidelity_update_all;
        
    case 'seperate SMS new'

        fidelity_update = image.*Data.sens_map;
        fidelity_update = fft2(fidelity_update,para.Recon.kSpace_size(1),para.Recon.kSpace_size(2));
        
        fidelity_update = sum(fidelity_update.*Data.SMS,5);
        fidelity_update = fidelity_update.*Data.mask;
        
        fidelity_update = Data.kSpace - fidelity_update;
        % norm for line search
        fidelity_norm = sum(abs(fidelity_update(:)).^2/prod(para.Recon.kSpace_size)/64);
        fidelity_norm = sqrt(fidelity_norm);
        % end
        fidelity_update = sum(fidelity_update.*conj(Data.SMS),7);

        if isfield(Data,'filter')
            fidelity_update = fidelity_update.*Data.filter;
        end
        fidelity_update = ifft2(fidelity_update);
        fidelity_update = sum(fidelity_update.*Data.sens_map_conj,4);
        return
    case 'seperate SMS new less memory'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = image.*Data.sens_map(:,:,:,i,:,:);
            fidelity_update = fft2(fidelity_update);
            fidelity_update = sum(fidelity_update.*Data.SMS,5);
            fidelity_update = fidelity_update.*Data.mask;
            fidelity_update = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update;
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)).^2/prod(para.Recon.kSpace_size)/64);
            fidelity_update = sum(fidelity_update.*conj(Data.SMS),7);
            if isfield(Data,'filter')
                fidelity_update = bsxfun(@times,fidelity_update,Data.filter);
            end
            fidelity_update = ifft2(fidelity_update);
            fidelity_update_all = fidelity_update_all + bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i,:,:));
        end
        fidelity_update = fidelity_update_all;
        fidelity_norm = sqrt(fidelity_norm);
        return

    case {'seperate SMS test','seperate SMS test less memory'}

        fidelity_update_all = single(zeros(size(image),class(image)));
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = image.*Data.sens_map(:,:,:,i,:,:);
            fidelity_update = fft2(fidelity_update);
            fidelity_update = sum(fidelity_update.*Data.SMS,5);
            siz = size(fidelity_update);
            fidelity_update_temp = zeros(siz,class(fidelity_update));
            fidelity_update = fidelity_update(Data.mask);
            fidelity_update = Data.kSpace(:,i) - fidelity_update;
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)).^2/prod(para.Recon.kSpace_size)/64);
            fidelity_update_temp(Data.mask) = fidelity_update;

            fidelity_update = fidelity_update_temp;clear fidelity_update_temp
            fidelity_update = sum(fidelity_update.*conj(Data.SMS),7);
            if isfield(Data,'filter')
                fidelity_update = bsxfun(@times,fidelity_update,Data.filter);
            end
            int_mask = sum(Data.mask,7);
            int_mask(int_mask==0) = 1;
            int_mask = 1./int_mask;
            fidelity_update = fidelity_update.*int_mask;
            
            fidelity_update = ifft2(fidelity_update);
            fidelity_update_all = fidelity_update_all + bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i,:,:));
        end
        fidelity_update = fidelity_update_all;
        fidelity_norm = sqrt(fidelity_norm);
        return
        
        
    case '3D SOS'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp

            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
%             fidelity_update = fftshift(fidelity_update,3);
            fidelity_update = fft(fidelity_update,[],3);
            kSpace_spiral = zeros(size(Data.kSpace), class(image));
            kSpace_spiral = kSpace_spiral(:,:,:,:,1);
            for iFrame = 1:size(fidelity_update,4)
%                 fprintf([num2str(iFrame),'\n'])
                for islice = 1:size(fidelity_update,3)
                    kSpace_spiral(:,:,islice,iFrame) = NUFFT.NUFFT(fidelity_update(:,:,islice,iFrame),Data.N(islice,iFrame));
                end
            end

            kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral .* Data.mask;
%             kSpace_spiral = kSpace_spiral.*Data.mask(:,:,:,:,i);
            fidelity_norm   = fidelity_norm + sum(abs(kSpace_spiral(:)).^2)/84/84/64;
            for iFrame = 1:size(fidelity_update,4)
%                 fprintf([num2str(iFrame),'\n'])
                for islice = 1:size(fidelity_update,3)
                    fidelity_update(:,:,islice,iFrame) = NUFFT.NUFFT_adj(kSpace_spiral(:,:,islice,iFrame),Data.N(islice,iFrame));
                end
            end
            fidelity_update = ifft(fidelity_update,[],3);
%             fidelity_update = fftshift(fidelity_update,3);

            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,:,i));
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        fidelity_update = fidelity_update_all;
        
    case '3D SOS server'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp

            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = fft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update, 3);
            kSpace_spiral = zeros(size(Data.kSpace), class(image));
            kSpace_spiral = kSpace_spiral(:,:,:,:,1);
            
            for islice = 1:size(fidelity_update,3)
                kSpace_spiral(:,:,islice,:) = NUFFT.NUFFT(permute(fidelity_update(:,:,islice,:),[1,2,4,3]),Data.N(islice));
            end

            kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral.*Data.mask;
            fidelity_norm = fidelity_norm + sum(abs(kSpace_spiral(:)).^2)/prod(Data.N(1).size_kspace);
            
            for islice = 1:size(fidelity_update,3)
                fidelity_update(:,:,islice,:) = NUFFT.NUFFT_adj(permute(kSpace_spiral(:,:,islice,:),[1,2,4,3]),Data.N(islice));
            end

            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = ifft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update,3);

            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,:,i));
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm);
        fidelity_update = fidelity_update_all;
        
    case '2D Spiral'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i));
            kSpace_spiral = zeros(size(Data.kSpace));
            kSpace_spiral = kSpace_spiral(:,:,:,1);
            for iFrame = 1:size(fidelity_update,3)
                kSpace_spiral(:,iFrame) = NUFFT.NUFFT(fidelity_update(:,:,iFrame),Data.N{iFrame});
            end

            kSpace_spiral = Data.kSpace(:,:,:,i) - kSpace_spiral;
            fidelity_norm = fidelity_norm + sum(abs(kSpace_spiral(:)).^2)/prod(Data.N.size_kspace);
            for iFrame = 1:size(fidelity_update,3)
                fidelity_update(:,:,iFrame) = NUFFT.NUFFT_adj(kSpace_spiral(:,iFrame),Data.N{iFrame});
            end

            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i));
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        fidelity_update = fidelity_update_all;
        
    case '2D Spiral server'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
            kSpace_spiral = NUFFT.NUFFT(fidelity_update,Data.N);
            kSpace_spiral = Data.kSpace(:,:,:,i,:) - kSpace_spiral;
            if isfield(Data, 'mask')
                kSpace_spiral = kSpace_spiral .* Data.mask;
            end
%             fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral)).^2)/prod(Data.N.size_kspace);
            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral.*(Data.N.W).^0.5)).^2)/prod(Data.N.size_kspace);
            
            fidelity_update = NUFFT.NUFFT_adj(kSpace_spiral,Data.N);
            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i,:));
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        fidelity_update = fidelity_update_all;
        
    case 'Toeplitz 2D'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;

        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
%             fidelity_update = bsxfun(@times,fidelity_update,Data.N.Apodizer);

            fidelity_update = fft2(fidelity_update,Data.N.size_kspace(1),Data.N.size_kspace(2));
            fidelity_update = bsxfun(@times,fidelity_update,Data.mask);
            fidelity_update = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update;
            
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
            clear fidelity_update_temp

            fidelity_update = ifft2(fidelity_update.*Data.mask);
            fidelity_update = fidelity_update(1:Data.N.size_image(1),1:Data.N.size_image(2),:,:);
            fidelity_update_all = fidelity_update_all + bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i,:));
        end
        fidelity_update = fidelity_update_all .* Data.N.Apodizer;
        fidelity_norm   = sqrt(fidelity_norm)/2;

        return
        
    case {'2D spiral'}

        fidelity_update = bsxfun(@times, image, Data.sens_map);
        fidelity_update = fft2(fidelity_update);
        fidelity_update = bsxfun(@times, fidelity_update, Data.mask);

        fidelity_update = Data.kSpace - fidelity_update;
        
%         fidelity_norm = fidelity_update/para.Recon.kSpace_size(1)/8;
        
                
        if isfield(Data,'filter')
            fidelity_update = fidelity_update .* Data.filter; % use filter to accelerate converge
        end
        
        fidelity_update = ifft2(fidelity_update);
        fidelity_update = bsxfun(@times, fidelity_update, Data.sens_map_conj);
        fidelity_update = sum(fidelity_update, 4);
        fidelity_norm = fidelity_update;
        
    case '2D Spiral Off Res Corr'
        
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i = 1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i));
            
            kSpace_spiral = zeros(size(Data.kSpace(:,:,:,i)), class(Data.kSpace));
            for j = 1:para.L
                kSpace_spiral = kSpace_spiral + NUFFT.NUFFT(fidelity_update .* Data.off_res.f_im(:, :, :, j), Data.N) .* Data.off_res.f_k(:, :, :, j);
            end
            kSpace_spiral = Data.kSpace(:,:,:,i) - kSpace_spiral;
            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral.*(Data.N.W).^0.5)).^2)/prod(Data.N.size_kspace);
            
            fidelity_update = zeros(size(fidelity_update), class(fidelity_update));
            for j = 1:para.L
                fidelity_update = fidelity_update + NUFFT.NUFFT_adj(kSpace_spiral .* conj(Data.off_res.f_k(:, :, :, j)), Data.N) .* conj(Data.off_res.f_im(:, :, :, j));
            end
            fidelity_update = bsxfun(@times, fidelity_update, Data.sens_map_conj(:,:,:,i));
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        fidelity_update = fidelity_update_all;
        
    case '2D spiral sms server'
        fidelity_update_all = zeros(size(image), class(image));
        fidelity_norm = zeros([1], class(image));
        for i = 1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
            kSpace_spiral = NUFFT.NUFFT(fidelity_update, Data.N);
            kSpace_spiral = kSpace_spiral .* Data.phase_mod;
            kSpace_spiral = sum(kSpace_spiral, 5);
            
            kSpace_spiral = Data.kSpace(:,:,:,i) - kSpace_spiral;
%             fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral)).^2)/prod(Data.N.size_kspace);
            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral.*(Data.N.W).^0.5)).^2)/prod(Data.N.size_kspace);
            
            fidelity_update = NUFFT.NUFFT_adj(kSpace_spiral .* Data.phase_mod_conj, Data.N);
            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i,:));
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        fidelity_update = fidelity_update_all;
        
    case '2D spiral sms cine'
        fidelity_update_all = zeros(size(image), class(image));
        fidelity_norm = 0;
        nframe = size(image, 3);
        nsms = size(image, 5);
        
        
        for i = 1:para.Recon.no_comp
            kSpace_spiral = zeros([Data.N.size_data(1:2), nframe, 1, nsms], class(image));
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
            for j = 1:nframe
                kSpace_spiral(:, :, j, :, :) = NUFFT.NUFFT(fidelity_update(:, :, j, :, :), Data.N);
            end
            kSpace_spiral = kSpace_spiral .* Data.phase_mod;
            kSpace_spiral = sum(kSpace_spiral, 5);
            
            kSpace_spiral = Data.kSpace(:,:,:,i) - kSpace_spiral;
%             fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral)).^2)/prod(Data.N.size_kspace);
            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral.*(Data.N.W).^0.5)).^2)/prod(Data.N.size_kspace);
            
            for j = 1:nframe
                fidelity_update(:, :, j, :, :) = NUFFT.NUFFT_adj(kSpace_spiral(:, :, j) .* Data.phase_mod_conj(:, :, j, :, :), Data.N);
            end
            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,i,:));
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        fidelity_update = fidelity_update_all;
        
    case '2D Sliding'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i = 1:para.Recon.no_comp
            
            image = reshape(image, [para.Recon.sx, para.Recon.sy, para.Recon.n_cardiac_phase, para.Recon.n_cardiac_cycle]);
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
            fidelity_update = reshape(fidelity_update, [para.Recon.sx, para.Recon.sy, para.Recon.n_frame]);
            
            kSpace_spiral = NUFFT.NUFFT(fidelity_update,Data.N);
            kSpace_spiral = Data.kSpace(:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral .* Data.mask;
%             fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral)).^2)/prod(Data.N.size_kspace);
            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral.*(Data.N.W).^0.5)).^2)/prod(Data.N.size_kspace);
            
            fidelity_update = NUFFT.NUFFT_adj(kSpace_spiral,Data.N);
            
            fidelity_update = reshape(fidelity_update, [para.Recon.sx, para.Recon.sy, para.Recon.n_cardiac_phase, para.Recon.n_cardiac_cycle]);
            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,:,i));
            fidelity_update = reshape(fidelity_update, [para.Recon.sx, para.Recon.sy, para.Recon.n_frame]);
            
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        fidelity_update = fidelity_update_all;
        
        
    case '3D SOS new'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i = 1:para.Recon.no_comp

            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = fft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update, 3);
            kSpace_spiral = zeros(size(Data.kSpace), class(image));
            kSpace_spiral = kSpace_spiral(:,:,:,:,1);
            
            for islice = 1:size(fidelity_update, 3)
                for ispiral = 1:size(kSpace_spiral, 2)
                    kSpace_spiral(:,ispiral,islice,:) = NUFFT.NUFFT(permute(fidelity_update(:,:,islice,:),[1,2,4,3]),Data.N(ispiral,islice));
                end
            end

            kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral .* Data.mask;
%             fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral .* (Data.N(1).W).^0.5)).^2) / prod(Data.N(1).size_kspace) / size(fidelity_update, 3);
            fidelity_norm = fidelity_norm + sum(abs(kSpace_spiral(:)).^2)/prod(Data.N(1).size_kspace) / 30;
            
            for islice = 1:size(fidelity_update, 3)
                fidelity_update_temp = zeros([Data.N(1).size_image, Data.N(1).size_data(3)], class(fidelity_update));
                for ispiral = 1:size(kSpace_spiral, 2)
                    fidelity_update_temp = fidelity_update_temp + NUFFT.NUFFT_adj(permute(kSpace_spiral(:,ispiral,islice,:),[1,2,4,3]),Data.N(ispiral, islice));
                end
                fidelity_update(:,:,islice,:) = fidelity_update_temp;
            end

            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = ifft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update,3);

            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,:,i));
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm);
        fidelity_update = fidelity_update_all;
        
    case '3D SOS new 2'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i = 1:para.Recon.no_comp

            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
            
            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = fft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update, 3);
            
            kSpace_spiral = zeros(size(Data.kSpace), class(image));
            kSpace_spiral = kSpace_spiral(:,:,:,:,1);
            
            for islice = 1:size(fidelity_update, 3)
                for iframe = 1:size(fidelity_update, 4)
                    kSpace_spiral(:,:,islice,iframe) = NUFFT.NUFFT(fidelity_update(:,:,islice,iframe),Data.N);
                end
            end

            kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral .* Data.mask;
            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral .* (Data.N(1).W).^0.5)).^2) / prod(Data.N(1).size_kspace) / size(fidelity_update, 3);
            
            for islice = 1:size(fidelity_update, 3)
%                 fidelity_update_temp = zeros([Data.N(1).size_image, Data.N(1).size_data(3)], class(fidelity_update));
                for iframe = 1:1:size(fidelity_update, 4)
                    fidelity_update(:,:,islice,iframe) = NUFFT.NUFFT_adj(kSpace_spiral(:,:,islice,iframe),Data.N);
                end
%                 fidelity_update(:,:,iSlice,iframe) = fidelity_update_temp;
            end

            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = ifft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update,3);

            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,:,i));
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm);
        fidelity_update = fidelity_update_all;
        
    case '3D SOS new 3'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        
        for i = 1:para.Recon.no_comp % coil

            fidelity_update = image .* Data.sens_map(:,:,:,:,i);
            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = fft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update, 3);
            
            kSpace_spiral = zeros(size(Data.kSpace), class(image));
            kSpace_spiral = kSpace_spiral(:,:,:,:,1);
            
            for islice = 1:size(kSpace_spiral, 3)
                for ispiral = 1:size(kSpace_spiral, 2)
                    mask_temp = Data.mask(:, ispiral, islice, :);
                    kSpace_spiral(:,ispiral,islice,mask_temp) = NUFFT.NUFFT(permute(fidelity_update(:,:,islice,mask_temp),[1,2,4,3]),Data.N(ispiral,islice));
                end
            end

            kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral .* Data.mask;
%             fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral .* (Data.N(1).W).^0.5)).^2) / prod(Data.N(1).size_kspace) / size(fidelity_update, 3);
            fidelity_norm = fidelity_norm + sum(abs(kSpace_spiral(:)).^2)/prod(Data.N(1).size_kspace);
            
            for islice = 1:size(fidelity_update, 3)
                fidelity_update_temp = zeros([Data.N(1).size_image, Data.N(1).size_data(3)], class(fidelity_update));
                for ispiral = 1:size(kSpace_spiral, 2)
                    mask_temp = Data.mask(:, ispiral, islice, :);
                    fidelity_update_temp(:,:,mask_temp) = fidelity_update_temp(:,:,mask_temp) + NUFFT.NUFFT_adj(permute(kSpace_spiral(:,ispiral,islice,mask_temp),[1,2,4,3]),Data.N(ispiral, islice));
                end
                fidelity_update(:,:,islice,:) = fidelity_update_temp;
            end

            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = ifft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update,3);

            fidelity_update = fidelity_update .* Data.sens_map_conj(:,:,:,:,i);
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm);
        fidelity_update = fidelity_update_all;
        
    case '2D new nufft'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i = 1:para.Recon.no_comp
            fidelity_update = image .* Data.sens_map(:,:,:,i);
            kSpace_spiral = nufft(fidelity_update,Data.N);
            kSpace_spiral = Data.kSpace(:,:,:,i) - kSpace_spiral;
            if isfield(Data, 'mask')
                kSpace_spiral = kSpace_spiral .* Data.mask;
            end

            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral.*(Data.N.W).^0.5)).^2)/prod(Data.N.size_kspace);
            
            fidelity_update = nufft_adj(kSpace_spiral,Data.N);
            fidelity_update = fidelity_update .* Data.sens_map_conj(:,:,:,i);
            fidelity_update_all = fidelity_update_all + fidelity_update;
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        fidelity_update = fidelity_update_all;
end

%fidelity_update = Data.first_est - fidelity_update;