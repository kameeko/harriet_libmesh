%plot error distribution and setup (vertical, for vs-qoi and vs-data)

close all;
set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');

fntsize = 28;
maxref = 3; %LF = 0

numSens = 3; %3, 5, or 10
qoitype = 5; %3, 5, or 7

dir_root = ['qoi',num2str(qoitype),'_sens',num2str(numSens),'/'];

switch qoitype
    case 3
        qoiptsx = [0.625 0.875 0.875 0.625]; %qoi3
        qoiptsy = [0.375 0.375 0.625 0.625]; %qoi3
    case 5
        qoiptsx = [0 5 5 0]; %qoi5
        qoiptsy = [0 0 1 1]; %qoi5
    case 7
        qoiptsx = [0.625 1.5 1.5 0.625]; %qoi7
        qoiptsy = [0.25 0.25 0.75 0.75]; %qoi7
end
switch numSens
    case 3
        dataptsx = [0.35 1.56 3.1];
        dataptsy = [0.35 0.61 0.5];
    case 5
        dataptsx = [0.35 0.98 1.56 2.46 3.1];
        dataptsy = [0.35 0.51 0.61 0.37 0.5];
    case 10
        dataptsx = [0.35 0.98 1.11 1.56 2.23 2.46 2.92 3.1 3.49 4.79];
        dataptsy = [0.35 0.51 0.89 0.61 0.51 0.37 0.7 0.5 0.76 0.26];
end

if exist('../fine_dof_info.mat','file') == 2
    load('../fine_dof_info.mat')
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

    save('../fine_dof_info.mat','dof_groups','elem_cent')
end

%plot setup
h = figure;
patch([0 0 1 1], [0 5 5 0],'w','edgecolor',[0 0 0],'edgealpha',1)
hold on
subreg = patch(qoiptsy, qoiptsx,...
  [0.87843137254902 0.690196078431373 1], ...
  'EdgeColor','none'); 
datapts = scatter(dataptsy,dataptsx,80, ...
  'MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor',[0 .7 .7]);
% axis equal
axis tight
xlim([0,1])
ylim([0,5])
xlabel('$x_2$');
ylabel('$x_1$');
set(gca,'XDir','Reverse')
set(gca,'PlotBoxAspectRatio',[0.5 1 1])
set(gca,'FontSize',fntsize);
set(findall(gcf,'type','text'),'FontSize',fntsize)
set(gcf,'PaperPositionMode','auto','Position',[620 95 504 668])
print(h,[dir_root,'setup_',num2str(qoitype),'_',num2str(numSens)],'-dpng','-r300')

% %plot error breakdowns
% for iter = 0:maxref
%     err_dof = dlmread(...
%         [dir_root,'error_est_breakdown_basis_blame',num2str(iter),'.dat'],' ');
%     
%     basis_errs = abs(sum(err_dof(dof_groups),2));
%     
%     %plot error breakdown
%     h = figure;
%     xs = unique(elem_cent(:,1));
%     xs = [0; 0.5*(xs(1:end-1)+xs(2:end)); 5];
%     ys = unique(elem_cent(:,2));
%     ys = [0; 0.5*(ys(1:end-1)+ys(2:end)); 1];
%     [x,y] = meshgrid(ys,xs);
%     surf(x,y,reshape(basis_errs,length(xs),length(ys)))
%     set(gca,'ytick',[])
%     shading interp
%     view(2)
% %     axis equal 
%     set(gca,'PlotBoxAspectRatio',[0.5 1 1])
%     axis tight
%     xlabel('$x_2$');
%     % ylabel('y');
%     c = colorbar; set(c,'TickLabelInterpreter','latex');
% %     title('Local Contributions to QoI Error Estimate','FontWeight','normal')
%     set(gca,'FontSize',fntsize);
%     set(findall(gcf,'type','text'),'FontSize',fntsize)
%     set(gca,'XDir','Reverse')
%     set(gcf,'PaperPositionMode','auto','Position',[620 95 504 668])
%     print(h,[dir_root, 'err_breakdown_', num2str(iter)],'-dpng','-r300')
% end