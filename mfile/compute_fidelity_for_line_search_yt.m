function fidelity_norm = compute_fidelity_for_line_search_yt(image, Data, para)
%--------------------------------------------------------------------------
%   [fidelity_norm] = compute_fidelity_yt_new(image, Data, para)
%--------------------------------------------------------------------------
%   Compute fidelity norm of a MRI reconstruction problem
%--------------------------------------------------------------------------
%   Inputs:      
%       - image             [sx, sy, nof, ...]
%       - Data              [structure]
%       - para              [structure]
%
%       - image             image
%       - Data              see 'help STCR_conjugate_gradient.m'
%       - para              see 'help STCR_conjugate_gradient.m'
%--------------------------------------------------------------------------
%   Output:
%       - fidelity_norm     [scalar]
%
%       - fidelity_norm     || Am - d ||_2^2
%--------------------------------------------------------------------------
%   A standard fidelity term it solves is:
%
%   || Am - d ||_2^2
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
    case 'seperate SMS less memory'
        fidelity_norm = 0;%zeros(size(image),class(image));

        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:,:));

            fidelity_update(para.Recon.image_size(1)+1:para.Recon.kSpace_size(1),para.Recon.image_size(2)+1:para.Recon.kSpace_size(2),:,:,:,:,:,:,:) = 0;
            fidelity_update = circshift(fidelity_update,[(para.Recon.kSpace_size(1)-para.Recon.image_size(1))/2,(para.Recon.kSpace_size(2)-para.Recon.image_size(2))/2]);
            fidelity_update = fft2(fidelity_update);
            
            fidelity_update_temp(:,:,:,1,1,:,1) = sum(fidelity_update,5);
            fidelity_update_temp(:,:,:,1,1,:,2) = fidelity_update(:,:,:,:,1,:,:) + exp(1i*2*pi/3)*fidelity_update(:,:,:,:,2,:,:) + exp(1i*4*pi/3)*fidelity_update(:,:,:,:,3,:,:);
            fidelity_update_temp(:,:,:,1,1,:,3) = fidelity_update(:,:,:,:,1,:,:) + exp(1i*4*pi/3)*fidelity_update(:,:,:,:,2,:,:) + exp(1i*2*pi/3)*fidelity_update(:,:,:,:,3,:,:);

            fidelity_update_temp = bsxfun(@times,fidelity_update_temp,Data.mask);
            fidelity_update_temp = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update_temp;
        

            %fidelity_update_temp = permute(fidelity_update_temp,[1,2,3,4,7,5,6]);
            %if isfield(Data,'filter')
            %    fidelity_update_temp = bsxfun(@times,fidelity_update_temp,Data.filter);
            %end
        

            %fidelity_update_all = fidelity_update_all + fidelity_update_temp/288/10;
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update_temp(:)/para.Recon.kSpace_size(1)/8).^2);

            clear fidelity_update_temp
        end
        fidelity_norm = sqrt(fidelity_norm);
    case 'seperate SMS'

        %fidelity_update = bsxfun(@times,image,Data.sens_map);
        fidelity_update = image.*Data.sens_map;

        fidelity_update(para.Recon.image_size(1)+1:para.Recon.kSpace_size(1),para.Recon.image_size(2)+1:para.Recon.kSpace_size(2),:,:,:,:,:,:,:) = 0;
        fidelity_update = circshift(fidelity_update,[(para.Recon.kSpace_size(1)-para.Recon.image_size(1))/2,(para.Recon.kSpace_size(2)-para.Recon.image_size(2))/2]);
        fidelity_update = fft2(fidelity_update);
            
        fidelity_update_temp(:,:,:,:,1,1,1) = sum(fidelity_update,5);
        fidelity_update_temp(:,:,:,:,1,1,2) = fidelity_update(:,:,:,:,1,:,:) + exp(1i*2*pi/3)*fidelity_update(:,:,:,:,2,:,:) + exp(1i*4*pi/3)*fidelity_update(:,:,:,:,3,:,:);
        fidelity_update_temp(:,:,:,:,1,1,3) = fidelity_update(:,:,:,:,1,:,:) + exp(1i*4*pi/3)*fidelity_update(:,:,:,:,2,:,:) + exp(1i*2*pi/3)*fidelity_update(:,:,:,:,3,:,:);
        
        clear fidelity_update
        
        %fidelity_update_temp = bsxfun(@times,fidelity_update_temp,Data.mask);
        fidelity_update_temp = fidelity_update_temp.*Data.mask;
        fidelity_update_temp = Data.kSpace - fidelity_update_temp;

        fidelity_norm = sum(abs(fidelity_update_temp(:)/para.Recon.kSpace_size(1)/8).^2);
        fidelity_norm = sqrt(fidelity_norm);

    case {'2D','2D less memory'}
        
        fidelity_norm = bsxfun(@times,image,Data.sens_map);
        fidelity_norm(para.Recon.image_size(1)+1:para.Recon.kSpace_size(1),para.Recon.image_size(2)+1:para.Recon.kSpace_size(2),:,:,:,:,:,:,:) = 0;
        fidelity_norm = circshift(fidelity_norm,[(para.Recon.kSpace_size(1)-para.Recon.image_size(1))/2,(para.Recon.kSpace_size(2)-para.Recon.image_size(2))/2]);
        fidelity_norm = fft2(fidelity_norm);
        fidelity_norm = bsxfun(@times,fidelity_norm,Data.mask);
%         fidelity_norm = (Data.kSpace - fidelity_norm)/para.Recon.kSpace_size(1)/8;
        fidelity_norm = (Data.kSpace - fidelity_norm) /para.Recon.kSpace_size(1);
        
    case 'MB5'
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
            fidelity_update = fft2(fidelity_update);
            fidelity_update_temp = sum(fidelity_update.*Data.phase,5);
            fidelity_update_temp = bsxfun(@times,fidelity_update_temp,Data.mask);
            fidelity_update_temp = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update_temp;
            
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update_temp(:)/para.Recon.kSpace_size(1)/8).^2);
            clear fidelity_update_temp
        end
        fidelity_norm = sqrt(fidelity_norm);
        
    case 'seperate SMS projections less memory'
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));

            %fidelity_update(para.Recon.image_size(1)+1:para.Recon.kSpace_size(1),para.Recon.image_size(2)+1:para.Recon.kSpace_size(2),:,:,:,:,:,:,:) = 0;
            %fidelity_update = circshift(fidelity_update,[(para.Recon.kSpace_size(1)-para.Recon.image_size(1))/2,(para.Recon.kSpace_size(2)-para.Recon.image_size(2))/2]);
            fidelity_update = fft2(fidelity_update);

            fidelity_update = fidelity_update.*Data.phase_mod;
            fidelity_update = sum(fidelity_update,5);

            fidelity_update = bsxfun(@times,fidelity_update,Data.mask);
            fidelity_update = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update;
        
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)/288/288).^2);
            
        end
        
    case '3D less memory'
        fidelity_norm = 0;

        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
            fidelity_update = fft3(fidelity_update);
            fidelity_update = bsxfun(@times,fidelity_update,Data.mask);
            fidelity_update = Data.kSpace(:,:,:,:,i) - fidelity_update;
        
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)/288/8).^2);
            
        end
        fidelity_norm = sqrt(fidelity_norm)/2;
        
    case '3D'
        fidelity_update = bsxfun(@times,image,Data.sens_map);
        fidelity_update = fft3(fidelity_update);
        fidelity_update = bsxfun(@times,fidelity_update,Data.mask);
        fidelity_update = Data.kSpace - fidelity_update;
        fidelity_norm = sum(abs(fidelity_update(:)/288/8).^2);
        fidelity_norm = sqrt(fidelity_norm)/2;
        
    case {'NUFFT', 'NUFFT RSG'}
        fidelity_update = bsxfun(@times,image,Data.sens_map);
        fidelity_update = NUFFT.NUFFT(fidelity_update,Data.N);
        if para.Recon.nSMS ~= 1
            fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod);
            fidelity_update = sum(fidelity_update,5);
            fidelity_update = Data.kSpace - fidelity_update;
            fidelity_norm = sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
        else
            fidelity_update = Data.kSpace - fidelity_update;
            fidelity_norm = sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
        end
        fidelity_norm = sqrt(fidelity_norm);
        
    case 'NUFFT coil'
        fidelity_update = NUFFT.NUFFT(image, Data.N);
        if para.Recon.nSMS ~= 1
            fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod);
            fidelity_update = sum(fidelity_update,5);
            fidelity_update = Data.kSpace - fidelity_update;
            fidelity_norm = sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
        else
            fidelity_update = Data.kSpace - fidelity_update;
            fidelity_norm = sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
        end
        fidelity_norm = sqrt(fidelity_norm);
        
    case 'NUFFT less memory'
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
            fidelity_update = NUFFT.NUFFT_new(fidelity_update,Data.N);
            if para.Recon.nSMS ~= 1
                fidelity_update = bsxfun(@times,fidelity_update,Data.phase_mod);
                fidelity_update = sum(fidelity_update,5);
                fidelity_update = Data.kSpace(:,:,:,i) - fidelity_update;
                fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)).^2)/288/288/64;
            else
                fidelity_update = Data.kSpace(:,:,:,i) - fidelity_update;
                fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)).^2)/288/288/64;
            end
        end
        fidelity_norm = sqrt(fidelity_norm);
        return
    case 'seperate SMS new'

        fidelity_update = image.*Data.sens_map;
        fidelity_update = fft2(fidelity_update,para.Recon.kSpace_size(1),para.Recon.kSpace_size(2));
        
        fidelity_update = sum(fidelity_update.*Data.SMS,5);
        fidelity_update = fidelity_update.*Data.mask;
        
        fidelity_update = Data.kSpace - fidelity_update;
        % norm for line search
        fidelity_norm = sum(abs(fidelity_update(:)).^2/prod(para.Recon.kSpace_size)/64);
        fidelity_norm = sqrt(fidelity_norm);

    case 'seperate SMS new less memory'
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = image.*Data.sens_map(:,:,:,i,:,:);
            fidelity_update = fft2(fidelity_update);
            fidelity_update = sum(fidelity_update.*Data.SMS,5);
            fidelity_update = fidelity_update.*Data.mask;
            fidelity_update = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update;
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)).^2/prod(para.Recon.kSpace_size)/64);
        end
        fidelity_norm = sqrt(fidelity_norm);
        return
        
    case {'seperate SMS test','seperate SMS test less memory'}
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = image.*Data.sens_map(:,:,:,i,:,:);
            fidelity_update = fft2(fidelity_update);
            fidelity_update = sum(fidelity_update.*Data.SMS,5);
            fidelity_update = fidelity_update(Data.mask);
            fidelity_update = Data.kSpace(:,i) - fidelity_update;
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)).^2/prod(para.Recon.kSpace_size)/64);
        end
        fidelity_norm = sqrt(fidelity_norm);
        return
        
    case '3D SOS'
        fidelity_update_all = zeros(size(image),class(image));
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp

            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
%             fidelity_update = fftshift(fidelity_update,3);
            fidelity_update = fft(fidelity_update,[],3);
            kSpace_spiral = zeros(size(Data.kSpace));
            kSpace_spiral = kSpace_spiral(:,:,:,:,1);
            for iFrame = 1:size(fidelity_update,4)
%                 fprintf([num2str(iFrame),'\n'])
                for iSlice = 1:size(fidelity_update,3)
                    kSpace_spiral(:,:,iSlice,iFrame) = NUFFT.NUFFT(fidelity_update(:,:,iSlice,iFrame),Data.N(iSlice,iFrame));
                end
            end

            kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral.*Data.mask;
            fidelity_norm   = fidelity_norm + sum(abs(kSpace_spiral(:)).^2)/84/84/64;
            %{
            for iFrame = 1:size(fidelity_update,4)
%                 fprintf([num2str(iFrame),'\n'])
                for iSlice = 1:size(fidelity_update,3)
                    fidelity_update(:,:,iSlice,iFrame) = NUFFT.NUFFT_adj(kSpace_spiral(:,:,iSlice,iFrame),Data.N{iSlice,iFrame});
                end
            end
            fidelity_update = ifft(fidelity_update,[],3);
%             fidelity_update = fftshift(fidelity_update,3);

            fidelity_update = bsxfun(@times,fidelity_update,Data.sens_map_conj(:,:,:,:,i));
            fidelity_update_all = fidelity_update_all + fidelity_update;
            %}

        end
        fidelity_norm = sqrt(fidelity_norm)/2;
%         fidelity_norm = fidelity_update_all;

    case '3D SOS server'
        fidelity_norm = 0;
        for i=1:para.Recon.no_comp

            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = fft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update, 3);
            kSpace_spiral = zeros(size(Data.kSpace), class(image));
            kSpace_spiral = kSpace_spiral(:,:,:,:,1);
            
            for iSlice = 1:size(fidelity_update,3)
                kSpace_spiral(:,:,iSlice,:) = NUFFT.NUFFT(permute(fidelity_update(:,:,iSlice,:),[1,2,4,3]),Data.N(iSlice));
            end

            kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral.*Data.mask;
            fidelity_norm = fidelity_norm + sum(abs(kSpace_spiral(:)).^2)/prod(Data.N(1).size_kspace);
        end
        fidelity_norm = sqrt(fidelity_norm);
        
    case '2D Spiral'
        
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
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        
    case '2D Spiral server'

        fidelity_norm = 0;
        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
            kSpace_spiral = NUFFT.NUFFT(fidelity_update,Data.N);
            kSpace_spiral = Data.kSpace(:,:,:,i,:) - kSpace_spiral;
            if isfield(Data, 'mask')
                kSpace_spiral = kSpace_spiral .* Data.mask;
            end
            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral.*(Data.N.W).^0.5)).^2)/prod(Data.N.size_kspace);
%             fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral)).^2)/prod(Data.N.size_kspace);
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        
    case 'Toeplitz 2D'

        fidelity_norm = 0;

        for i=1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
%             fidelity_update = bsxfun(@times,fidelity_update,Data.N.Apodizer);
            fidelity_update = fft2(fidelity_update,Data.N.size_kspace(1),Data.N.size_kspace(2));

            fidelity_update = bsxfun(@times,fidelity_update,Data.mask);
            fidelity_update = Data.kSpace(:,:,:,i,:,:,:) - fidelity_update;
            
            fidelity_norm = fidelity_norm + sum(abs(fidelity_update(:)).^2)/prod(Data.N.size_kspace);
        
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        
        
     case {'2D spiral'}

        fidelity_update = bsxfun(@times, image, Data.sens_map);
        fidelity_update = fft2(fidelity_update);
        fidelity_update = bsxfun(@times, fidelity_update, Data.mask);

        fidelity_update = Data.kSpace - fidelity_update;
                
        if isfield(Data,'filter')
            fidelity_update = fidelity_update .* Data.filter; % use filter to accelerate converge
        end
        
        fidelity_update = ifft2(fidelity_update);
        fidelity_update = bsxfun(@times, fidelity_update, Data.sens_map_conj);
        fidelity_update = sum(fidelity_update, 4);
        fidelity_norm = fidelity_update;
        
    case '2D Spiral Off Res Corr'
        
        fidelity_norm = 0;
        for i = 1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i));
            kSpace_spiral = zeros(size(Data.kSpace(:,:,:,i)), class(Data.kSpace));
            for j = 1:para.L
                kSpace_spiral = kSpace_spiral + NUFFT.NUFFT(fidelity_update .* Data.off_res.f_im(:, :, :, j), Data.N) .* Data.off_res.f_k(:, :, :, j);
            end
            kSpace_spiral = Data.kSpace(:,:,:,i) - kSpace_spiral;
            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral.*(Data.N.W).^0.5)).^2)/prod(Data.N.size_kspace);
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        
    case '2D spiral sms server'
%         fidelity_update_all = zeros(size(image), class(image));
        fidelity_norm = 0;
        for i = 1:para.Recon.no_comp
            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,i,:));
            kSpace_spiral = NUFFT.NUFFT(fidelity_update, Data.N);
            kSpace_spiral = kSpace_spiral .* Data.phase_mod;
            kSpace_spiral = sum(kSpace_spiral, 5);
            
            kSpace_spiral = Data.kSpace(:,:,:,i) - kSpace_spiral;
%             fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral)).^2)/prod(Data.N.size_kspace);
            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral.*(Data.N.W).^0.5)).^2)/prod(Data.N.size_kspace);

        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
%         fidelity_update = fidelity_update_all;

    case '2D spiral sms cine'
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
            
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        
    case '2D Sliding'
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
            
        end
        fidelity_norm   = sqrt(fidelity_norm)/2;
        
     case '3D SOS new'
        fidelity_norm = 0;
        for i = 1:para.Recon.no_comp

            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = fft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update, 3);
            kSpace_spiral = zeros(size(Data.kSpace), class(image));
            kSpace_spiral = kSpace_spiral(:,:,:,:,1);
            
            for iSlice = 1:size(fidelity_update, 3)
                for ispiral = 1:size(kSpace_spiral, 2)
                    kSpace_spiral(:,ispiral,iSlice,:) = NUFFT.NUFFT(permute(fidelity_update(:,:,iSlice,:),[1,2,4,3]),Data.N(ispiral,iSlice));
                end
            end

            kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral .* Data.mask;
%             fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral .* (Data.N(1).W).^0.5)).^2) / prod(Data.N(1).size_kspace) / size(fidelity_update, 3);
            fidelity_norm = fidelity_norm + sum(abs(kSpace_spiral(:)).^2)/prod(Data.N(1).size_kspace) / 30;
        end
        fidelity_norm   = sqrt(fidelity_norm);
        
     case '3D SOS new 2'
        fidelity_norm = 0;
        for i = 1:para.Recon.no_comp

            fidelity_update = bsxfun(@times,image,Data.sens_map(:,:,:,:,i));
            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = fft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update, 3);
            kSpace_spiral = zeros(size(Data.kSpace), class(image));
            kSpace_spiral = kSpace_spiral(:,:,:,:,1);
            
            for iSlice = 1:size(fidelity_update, 3)
                for iframe = 1:size(fidelity_update, 4)
                    kSpace_spiral(:,:,iSlice,iframe) = NUFFT.NUFFT(fidelity_update(:,:,iSlice,iframe),Data.N);
                end
            end

            kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral .* Data.mask;
            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral .* (Data.N(1).W).^0.5)).^2) / prod(Data.N(1).size_kspace) / size(fidelity_update, 3);
        end
        fidelity_norm   = sqrt(fidelity_norm);
        
        
    case '3D SOS new 3'
        fidelity_norm = 0;
        
        for i = 1:para.Recon.no_comp % coil
            fidelity_update = image .* Data.sens_map(:,:,:,:,i);
            fidelity_update = fftshift(fidelity_update, 3);
            fidelity_update = fft(fidelity_update,[],3);
            fidelity_update = fftshift(fidelity_update, 3);
            
            kSpace_spiral = zeros(size(Data.kSpace), class(image));
            kSpace_spiral = kSpace_spiral(:,:,:,:,1);
            
            for islice = 1:size(fidelity_update, 3)
                for ispiral = 1:size(kSpace_spiral, 2)
                    mask_temp = Data.mask(:, ispiral, islice, :);
                    kSpace_spiral(:,ispiral,islice,mask_temp) = NUFFT.NUFFT(permute(fidelity_update(:,:,islice,mask_temp),[1,2,4,3]),Data.N(ispiral,islice));
                end
            end

            kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
            kSpace_spiral = kSpace_spiral .* Data.mask;
%             fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral .* (Data.N(1).W).^0.5)).^2) / prod(Data.N(1).size_kspace) / size(fidelity_update, 3);
            fidelity_norm = fidelity_norm + sum(abs(kSpace_spiral(:)).^2)/prod(Data.N(1).size_kspace);
            
        end
        
    case '2D new nufft'
        fidelity_norm = 0;
        for i = 1:para.Recon.no_comp
            fidelity_update = image .* Data.sens_map(:,:,:,i);
            kSpace_spiral = nufft(fidelity_update,Data.N);
            kSpace_spiral = Data.kSpace(:,:,:,i) - kSpace_spiral;
            if isfield(Data, 'mask')
                kSpace_spiral = kSpace_spiral .* Data.mask;
            end

            fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral.*(Data.N.W).^0.5)).^2)/prod(Data.N.size_kspace);

        end
        fidelity_norm   = sqrt(fidelity_norm)/2;

        
end