%plot inferred fields
close all;

dir_root = 'long_channel_stash/qoi3_setup02_r4p2_deref/MF10/';
% dir_root = 'all_field/';
refLevel = 'MF10';

A = dlmread([dir_root,'f_',refLevel,'.csv'],',');
A = unique(A,'rows');
A = sortrows(A,[2 3]);

fntsize = 20;

meep = figure(1);
scatter(A(:,2),A(:,3),100,A(:,1),'s','filled');
xlabel('x'); 
ylabel('y'); 
title('Inferred f(q) (60% HF)','FontWeight','normal')
set(gca,'FontSize',fntsize); 
colorbar
caxis([-0.05 0.45])
set(findall(gcf,'type','text'),'FontSize',fntsize)
set(gcf,'PaperPositionMode','auto','Position',[66 253 1535 297])
print(meep,[dir_root, 'f_', refLevel],'-depsc');