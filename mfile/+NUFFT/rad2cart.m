function kSpace_cart = rad2cart(kSpace_non_cart, N)
%--------------------------------------------------------------------------
%   [kSpace_cart] = rad2cart(kSpace_cart, N)
%--------------------------------------------------------------------------
%   Interpolate non-Cartesian k-space on a Cartesian grid
%--------------------------------------------------------------------------
%   Inputs:
%       - kSpace_non_cart   [nsamples, nrep, nof, ...]
%       - N                 [structure]
%
%           'nsamples'  number of samples per readout (a radial ray or
%                       spiral arm)
%           'nrep'      number of repetetions per time frame (number of 
%                       radial rays or spiral arms)
%           'nof'       number of time frames
%
%       - kSpace_non_cart   non-Cartesian k-space data
%       - N                 see 'help NUFFT.init.m'
%--------------------------------------------------------------------------
%   Output:
%       - kSpace_cart       [sx, sx, nof, ...]
%                           inverse NUFFT result (not nessessary an image)
%--------------------------------------------------------------------------
%   Reference:
%       [1] Nonuniform Fast Fourier Transforms Using Min-Max Interpolation.
%           IEEE T-SP, 2003, 51(2):560-74. 
%--------------------------------------------------------------------------
%   Author:
%       Ye Tian
%       E-mail: phye1988@gmail.com
%--------------------------------------------------------------------------
sx    	= size(kSpace_non_cart,1);
nor     = size(kSpace_non_cart,2);
nof     = size(kSpace_non_cart,3);
nc      = size(kSpace_non_cart,4);
nSMS    = size(kSpace_non_cart,5);
ns      = size(kSpace_non_cart,6);
sx_over = N.sx_over;

kSpace_non_cart = reshape(kSpace_non_cart, [sx*nor*nof,nc*nSMS*ns]);

kSpace_cart = single(N.S * double(kSpace_non_cart));

kSpace_cart = reshape(kSpace_cart,[sx_over, sx_over, nof, nc, nSMS, ns]);
