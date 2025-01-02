function showImage3D(Image,Cost)
Image = crop_half_FOV(abs(Image));
[nx,ny,nz,nof] = size(Image);
frame_num = floor(nof/4);
if frame_num ~= 0
    im = Image(:,:,:,[frame_num frame_num*2 frame_num*3]);
    im = permute(im,[1 4 2 3]);
    im = reshape(im,[nx*3 ny*nz]);
else
    im = Image(:,:,round(nz/9):round(nz/9):nz);
    im = squeeze(im);
    im = reshape(im,[nx,ny,2,4]);
    im = permute(im,[1,3,2,4]);
    im = reshape(im,[nx*2,ny*4]);
end

figure(1)
subplot(1,2,1)
imagesc(im)
colormap gray
brighten(0.3)
axis image
axis off
        
subplot(1,2,2)
plotCost(Cost)
drawnow
end
