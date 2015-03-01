%to make pretty plots from error breakdown file

dir_root = 'with_reaction/long_channel_stash/qoi5_setup02_r42/MF08/';

errFileID = ...
  fopen([dir_root,'error_est_breakdown.dat'],'r');
errFormatSpec = '%f %f %f';
errSizeMat = [3 Inf];

A = fscanf(errFileID,errFormatSpec,errSizeMat);
A = A';

figure(1); 
scatter(A(:,1),A(:,2),300,A(:,3),'s','filled'); colorbar
xlabel('x'); ylabel('y'); 
title('QoI Error Estimate Contribution')
set(gca,'FontSize',20); 
set(findall(gcf,'type','text'),'FontSize',20)
caxis([0 0.001])
set(gcf,'Position',[66 253 1535 297])

divFileID = ...
  fopen([dir_root,'divvy.txt'],'r');
divFormatSpec = '%d %d';
divSizeMat = [2 Inf];

B = fscanf(divFileID,divFormatSpec,divSizeMat);
B = B';
B = B(:,2);

LFbits = (B == 0);
HFbits = (B == 1);

figure(2); hold on;
scatter(A(LFbits,1),A(LFbits,2),300,[227, 38, 54]/255,'s','filled');
scatter(A(HFbits,1),A(HFbits,2),300,[91, 146, 229]/255,'s','filled');
xlabel('{ }'); ylabel('{ }'); 
title('{ }')
set(gca,'FontSize',20); 
set(findall(gcf,'type','text'),'FontSize',20)
set(gcf,'Position',[66 253 1475 297])
