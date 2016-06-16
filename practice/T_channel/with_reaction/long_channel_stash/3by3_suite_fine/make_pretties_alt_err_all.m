%plot error distribution and domain division for alternate error breakdown

%save inverse map to mat file? toggle option for coarser or finer version

close all;
set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');

fntsize = 32;
finer = true; %true = 250x50, false = 75x15

refLevel = 'LF';
% dir_root = ['qoi3_sens3_alt_err_all/',refLevel,'/'];
dir_root = 'vs_qoi_alt_err_all/qoi3_sens3/';

if finer
    if exist('fine_dof_info.mat','file') == 2
        load('fine_dof_info.mat')
    else
        dof_global = dlmread('../../psi_and_superadj/global_dof_map.dat',' ');
        elem_cent = dlmread('../../psi_and_superadj/elem_centroids.dat',' ');
        
        %remove weird 0's at end, and redundant indices at beginning
        dof_global = dof_global(:,2:end-1);
        
        maxdof = max(max(dof_global(:,2:end)));
        
        %inverse map - which dofs have which elements in support
        inv_dof_global = zeros(maxdof+1,ceil(length(dof_global(:))/(maxdof+1)));
        for dof = 0:maxdof
            [row, ~] = find(dof_global == dof);
            inv_dof_global(dof+1,1:length(row)) = row;
        end
        
        %group global dofs which have same elements in support
        [dof_group_elems,m,n] = unique(inv_dof_global,'rows');
        dof_groups = zeros(size(dof_group_elems,1),6);
        for ii = 1:size(dof_groups,1)
            dof_groups(ii,:) = find(n == ii);
        end
        
        save('fine_dof_info.mat','dof_groups','elem_cent')
    end
else
    if exist('coarse_dof_info.mat','file') == 2
       load('coarse_dof_info.mat')
    else
        dof_global = dlmread('../../psi_and_superadj/global_dof_map.dat',' ');
        elem_cent = dlmread('../../psi_and_superadj/elem_centroids.dat',' ');
        
        %remove weird 0's at end, and redundant indices at beginning
        dof_global = dof_global(:,2:end-1);
        
        maxdof = max(max(dof_global(:,2:end)));
        
        %inverse map - which dofs have which elements in support
        inv_dof_global = zeros(maxdof+1,ceil(length(dof_global(:))/(maxdof+1)));
        for dof = 0:maxdof
            [row, ~] = find(dof_global == dof);
            inv_dof_global(dof+1,1:length(row)) = row;
        end
        
        %group global dofs which have same elements in support
        [dof_group_elems,m,n] = unique(inv_dof_global,'rows');
        dof_groups = zeros(size(dof_group_elems,1),size(dof_var,1));
        for ii = 1:size(dof_groups,1)
            dof_groups(ii,:) = find(n == ii);
        end
        
        save('coarse_dof_info.mat','dof_groups','elem_cent')
    end
end

%read in things to plot
err_dof = dlmread([dir_root,'error_est_breakdown_basis_blame3.dat'],' ');
divFileID = ...
  fopen([dir_root,'divvy.txt'],'r');
divFormatSpec = '%d %d';
divSizeMat = [2 Inf];

%% plot error breakdown

basis_errs = abs(sum(err_dof(dof_groups),2));

%plot error breakdown
h = figure;
xs = unique(elem_cent(:,1));
xs = [0; 0.5*(xs(1:end-1)+xs(2:end)); 5];
ys = unique(elem_cent(:,2));
ys = [0; 0.5*(ys(1:end-1)+ys(2:end)); 1];
[x,y] = meshgrid(xs,ys);
surf(x,y,reshape(basis_errs,length(xs),length(ys)).')
set(gca,'ytick',[])
shading interp
view(2)
axis equal tight
xlabel('$x_1$'); 
% ylabel('y'); 
c = colorbar; set(c,'TickLabelInterpreter','latex');
title('Local Contributions to QoI Error Estimate','FontWeight','normal')
set(gca,'FontSize',fntsize); 
set(findall(gcf,'type','text'),'FontSize',fntsize)
% set(gcf,'PaperPositionMode','auto','Position',[66 253 1535 297])
% set(gcf,'PaperPositionMode','auto','Position',[66 140 1535 407])
set(gcf,'PaperPositionMode','auto','Position',[88 202 1336 409])
% print(h,[dir_root, 'err_breakdown_', refLevel],'-depsc');
print(h,[dir_root, 'err_breakdown_', refLevel],'-dpng','-r300')

% %% plot divvy
% 
% if divFileID ~= -1
%     B = fscanf(divFileID,divFormatSpec,divSizeMat);
%     B = B';
%     B = B(:,2);
% else
%     if finer
%         B = zeros(50*250,1);
%     else
%         B = zeros(15*75,1);
%     end
% end
% 
% LFbits = (B == 0);
% HFbits = (B == 1);
% 
% divvy = figure(2); 
% % hold on;
% % scatter(A(LFbits,1),A(LFbits,2),270,[227, 38, 54]/255,'s','filled');
% % scatter(A(HFbits,1),A(HFbits,2),270,[91, 146, 229]/255,'s','filled');
% map = [91, 146, 229; 227, 38, 54]/255;
% squish = reshape(B,5*round(sqrt(length(HFbits)/5)),round(sqrt(length(HFbits)/5)))';
% imagesc([0 5],[0 1],-squish); 
% set(gca,'YDir','normal','Ticklength',[0 0]); colormap(map);
% xlabel('$x_1$'); ylabel('$x_2$'); axis equal; axis tight;
% title('Division of Domain','FontWeight','Normal')
% set(gca,'FontSize',fntsize); 
% set(findall(gcf,'type','text'),'FontSize',fntsize)
% % set(gcf,'PaperPositionMode','auto','Position',[66 253 1475 297])
% % set(gcf,'PaperPositionMode','auto','Position',[66 253 1150 297])
% set(gcf,'PaperPositionMode','auto','Position',[66 218 1160 365])
% % print(divvy,[dir_root, 'cd_cdr_', refLevel, '_divvy'],'-depsc');
% print(divvy,[dir_root, 'cd_cdr_', refLevel, '_divvy'],'-dpng','-r300')