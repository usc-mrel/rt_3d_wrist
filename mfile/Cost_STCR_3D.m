function Cost_new = Cost_STCR_3D(fUpdate, Image, sWeight, tWeight, slWeight, Cost_old)

N = numel(Image);

fNorm = sum(abs(fUpdate(:)).^2)/N;

if tWeight ~= 0
    tNorm = abs(diff(Image,1,4));
    tNorm = tWeight * sum(tNorm(:))/N;
else
    tNorm = 0;
end

if sWeight ~= 0
    sx_norm = abs(diff(Image,1,1));
    sx_norm(end+1,:,:,:,:)=0;
    sy_norm = abs(diff(Image,1,2));
    sy_norm(:,end+1,:,:,:)=0;
    %sz_norm = abs(diff(Image,1,3));
    %sz_norm(:,:,end+1,:,:)=0;
    %sNorm = sqrt(abs(sx_norm).^2+abs(sy_norm).^2+abs(sz_norm).^2);
    sNorm = sqrt(abs(sx_norm).^2+abs(sy_norm).^2);
    sNorm = sWeight * sum(sNorm(:))/N;
else
    sNorm = 0;
end

if slWeight ~= 0 
    zNorm = abs(diff(Image,1,3));
    zNorm = slWeight * sum(zNorm(:))/N;
else
    zNorm = 0;
end

sNorm = sNorm + zNorm;

Cost = sNorm + tNorm + fNorm;

if nargin == 5
    Cost_new = Cost;
    return
end

Cost_new = Cost_old;

if isempty(Cost_old.fidelityNorm)==1
    Cost_new.fidelityNorm = gather(fNorm);
    Cost_new.temporalNorm = gather(tNorm);
    Cost_new.spatialNorm = gather(sNorm);
    Cost_new.totalCost = gather(Cost);
else
    Cost_new.fidelityNorm(end+1) = gather(fNorm);
    Cost_new.temporalNorm(end+1) = gather(tNorm);
    Cost_new.spatialNorm(end+1) = gather(sNorm);
    Cost_new.totalCost(end+1) = gather(Cost);
end

end