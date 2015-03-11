%to make pretty plots from error breakdown file
close all;

refLevel = 'MF08';
dir_root = ['',...
  'diff_param_res/with_reaction/long_channel_stash/'...
  'qoi6_setup02_r4p2/',refLevel,'/'];

errFileID = ...
  fopen([dir_root,'error_est_breakdown.dat'],'r');
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
caxis([0 0.001])
set(gcf,'PaperPositionMode','auto','Position',[66 253 1535 297])
print(err,[dir_root, 'err_breakdown_', refLevel,'_v2'],'-depsc');
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

divvy = figure(2); hold on;
scatter(A(LFbits,1),A(LFbits,2),270,[227, 38, 54]/255,'s','filled');
scatter(A(HFbits,1),A(HFbits,2),270,[91, 146, 229]/255,'s','filled');
xlabel('{ }'); ylabel('{ }'); 
title('{ }')
set(gca,'FontSize',20); 
set(findall(gcf,'type','text'),'FontSize',20)
set(gcf,'PaperPositionMode','auto','Position',[66 253 1475 297])
print(divvy,[dir_root, 'cd_cdr_', refLevel, '_divvy'],'-depsc');
