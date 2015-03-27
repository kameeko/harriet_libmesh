%to make pretty plots from error breakdown file
close all;

refLevel = 'MF06';
dir_root = ['',...
  'with_reaction/long_channel_stash/'...
  '3by3_suite/qoi3_sens3/',refLevel,'/'];

errFileID = ...
  fopen([dir_root,'error_est_breakdown_beta.dat'],'r');
errFormatSpec = '%f %f %f';
errSizeMat = [3 Inf];

A = fscanf(errFileID,errFormatSpec,errSizeMat);
A = A';

err = figure(1); 
scatter(A(:,1),A(:,2),300,A(:,3),'s','filled'); colorbar
% hold on
% patch([0.625 1.5 1.5 0.625],...
%   [0.25 0.25 0.75 0.75],'w',...
%   'EdgeColor',[0.87843137254902 0.690196078431373 1],...
%   'LineWidth',5,'FaceColor','none');
% scatter([0.35 1.56 3.1],[0.35 0.61 0.5],100, ...
%   'MarkerEdgeColor','w',...
%   'LineWidth',3,...
%   'MarkerFaceColor',[0 .7 .7]);
xlabel('x'); ylabel('y'); 
title('QoI Error Estimate Contribution')
% title('{ }')
set(gca,'FontSize',20); 
set(findall(gcf,'type','text'),'FontSize',20)
caxis([0 0.00004])
set(gcf,'PaperPositionMode','auto','Position',[66 253 1535 297])
print(err,[dir_root, 'err_breakdown_', refLevel,'_beta'],'-depsc');

divFileID = ...
  fopen([dir_root,'divvy.txt'],'r');
divFormatSpec = '%d %d';
divSizeMat = [2 Inf];

if divFileID ~= -1
  B = fscanf(divFileID,divFormatSpec,divSizeMat);
  B = B';
  B = B(:,2);
else
  B = zeros(size(A,1),1);
end

LFbits = (B == 0);
HFbits = (B == 1);

divvy = figure(2); 
% hold on;
% scatter(A(LFbits,1),A(LFbits,2),270,[227, 38, 54]/255,'s','filled');
% scatter(A(HFbits,1),A(HFbits,2),270,[91, 146, 229]/255,'s','filled');
map = [91, 146, 229; 227, 38, 54]/255;
squish = reshape(B,75,15)';
imagesc([0 5],[0 1],-squish); 
set(gca,'YDir','normal','Ticklength',[0 0]); colormap(map);
xlabel('{ }'); ylabel('{ }'); 
title('{ }')
set(gca,'FontSize',20); 
set(findall(gcf,'type','text'),'FontSize',20)
set(gcf,'PaperPositionMode','auto','Position',[66 253 1475 297])
print(divvy,[dir_root, 'cd_cdr_', refLevel, '_divvy'],'-depsc');
