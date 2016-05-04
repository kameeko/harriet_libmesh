%easier way to do refinement...

close all;

refPortion = 0.2; %total, not in addition to previous
    %portion of basis functions; portion of elements will be slightly off
fntsize = 16;

err_dof = dlmread('error_est_breakdown_basis_blame.dat',' ');
dof_global = dlmread('global_dof_map.dat',' ');
dof_var = dlmread('var_dof_map.dat',' ');
elem_cent = dlmread('elem_centroids.dat',' ');
divvy_prev = 'divvy.txt'; %current state of refinement

%remove weird 0's at end, and redundant indices at beginning
dof_var = dof_var(:,2:end-1); 
dof_global = dof_global(:,2:end-1);

maxdof = max(max(dof_var(:,2:end)));

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

%combine error
basis_errs = abs(sum(err_dof(dof_groups),2));

%find current refinement level
divFileID = fopen(divvy_prev,'r');
divFormatSpec = '%d %d';
divSizeMat = [2 Inf];
if(divFileID >= 3)
    divvy = fscanf(divFileID,divFormatSpec,divSizeMat);
    divvy = divvy';
    divvy = divvy(:,2);
else %no previous refinement
    divvy = zeros(size(elem_cent,1),1);
end
curr_ref = sum(divvy)/length(divvy);
add_ref = refPortion - curr_ref; %additional refinement portion
if(add_ref < 0)
    disp('Aaah already refined that much...')
end

%tag worst offenders
nBasis = size(basis_errs,1);
cutoff = round((1-add_ref)*nBasis);

meep = sort(basis_errs);
bloop = (basis_errs >= meep(cutoff));

%tag elements in support of worst offenders
eep = dof_group_elems(bloop,:);
eep = unique(eep(:));
eep(eep == 0) = [];

%write file to make next mesh assignment
newref = zeros(size(divvy));
newref(eep) = 1;
divvynew = newref + divvy;
divvynew(divvynew == 2) = 1;
divvynew = [(0:1:(length(divvy)-1))' divvynew];
dlmwrite('do_divvy.txt',divvynew,' ')

sum(divvynew(:,2))/length(divvynew(:,1))

%plot error breakdown
%dof groups seem to already be ordered left to right, bottom to top...
h = figure;
xs = unique(elem_cent(:,1));
xs = [0; 0.5*(xs(1:end-1)+xs(2:end)); 5];
ys = unique(elem_cent(:,2));
ys = [0; 0.5*(ys(1:end-1)+ys(2:end)); 1];
[x,y] = meshgrid(xs,ys);
surf(x,y,reshape(basis_errs,length(xs),length(ys)).')
shading interp
view(2)
% axis equal tight
xlabel('x'); ylabel('y'); colorbar;
title('Elemental Decomposition of QoI Error Estimate','FontWeight','normal')
set(gca,'FontSize',fntsize); 
set(findall(gcf,'type','text'),'FontSize',fntsize)
set(gcf,'PaperPositionMode','auto','Position',[66 253 1535 297])
% print(h,'err_breakdown','-depsc');
print(h,'err_breakdown','-dpng','-r300')

%plot new domain division
figure;
squish = reshape(divvynew(:,2),250,50)';
map = [91, 146, 229; 227, 38, 54]/255;
imagesc([0 5],[0 1],-squish); 
set(gca,'YDir','normal','Ticklength',[0 0]); colormap(map);
xlabel('x'); ylabel('y'); 
title('Division of Domain','FontWeight','normal')
set(gca,'FontSize',fntsize); 
set(findall(gcf,'type','text'),'FontSize',fntsize)
set(gcf,'PaperPositionMode','auto','Position',[66 253 1475 297])
hold on; 
contour(reshape(elem_cent(:,1),250,50),...
    reshape(elem_cent(:,2),250,50),...
    reshape(divvy,250,50),[0 1],'k','LineWidth',1)
