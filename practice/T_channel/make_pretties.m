%to make pretty plots from error breakdown file
close all;

refLevel = 'MF03';
dir_root = ['',...
  'with_reaction/long_channel_stash/'...
  '3by3_suite_fine/qoi3_sens3/',refLevel,'/'];

errFileID = ...
  fopen([dir_root,'error_est_breakdown.dat'],'r');
errFormatSpec = '%f %f %f';
errSizeMat = [3 Inf];

A = fscanf(errFileID,errFormatSpec,errSizeMat);
A = A';

err = figure(1); 
scatter(A(:,2),A(:,1),100,A(:,3),'s','filled'); colorbar
% scatter(A(:,1),A(:,2),100,A(:,3),'s','filled'); colorbar
% scatter(A(:,1),A(:,2),300,A(:,3),'s','filled'); colorbar
% hold on
% patch([0.625 1.5 1.5 0.625],...
%   [0.25 0.25 0.75 0.75],'w',...
%   'EdgeColor',[0.87843137254902 0.690196078431373 1],...
%   'LineWidth',5,'FaceColor','none');
% scatter([0.35 1.56 3.1],[0.35 0.61 0.5],100, ...
%   'MarkerEdgeColor','w',...
%   'LineWidth',3,...
%   'MarkerFaceColor',[0 .7 .7]);
% xlabel('x'); 
% ylabel('y'); 
% set(gca,'xtick',[])
% set(gca,'YTickMode','Manual','YTickLabelMode','Manual')
set(gca,'ytick',[])
ylim([0 5])
% set(gca,'xtickLabel',{' ',' ',' '})
xlabel('y','FontSize',14)
set(gca,'XDir','Reverse')
% set(gca,'ytickLabel',{' ',' ',' '})
% xlabel('{ }'); 
% ylabel('{ }'); 
% title('Elemental Decomposition of QoI Error Estimate')
% title('QoI Error Estimate Contribution')
% title('{ }')
set(gca,'FontSize',14); 
% meep=colorbar;
caxis([0 0.000037])
% set(meep,'FontSize',15)
% set(findall(gcf,'type','text'),'FontSize',20)
% set(gcf,'PaperPositionMode','auto','Position',[66 253 1535 297])
% set(gcf,'PaperPositionMode','auto','Position',[66 253 1200 297])
% set(gcf,'PaperPositionMode','auto','Position',[66 253 1200 333])
set(gcf,'PaperPositionMode','auto','Position',[1 1 300/1.5 1150/4.1])
% set(gcf,'PaperPositionMode','auto','Position',[1 1 300/1.5 1150/3.6])
print(err,[dir_root, 'err_breakdown_', refLevel],'-depsc');
break
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
% squish = reshape(B,75,15)';
squish = reshape(B,250,50)';
imagesc([0 5],[0 1],-squish); 
set(gca,'YDir','normal','Ticklength',[0 0]); colormap(map);
xlabel('x'); ylabel('y'); 
title('Division of Domain')
set(gca,'FontSize',32); 
set(findall(gcf,'type','text'),'FontSize',32)
% set(gcf,'PaperPositionMode','auto','Position',[66 253 1475 297])
% set(gcf,'PaperPositionMode','auto','Position',[66 253 1150 297])
set(gcf,'PaperPositionMode','auto','Position',[66 253 1150 333])
print(divvy,[dir_root, 'cd_cdr_', refLevel, '_divvy'],'-depsc');
