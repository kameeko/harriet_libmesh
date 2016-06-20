%to create "setup" plots (version for paper)
close all;

close all;
set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');

fntsize = 28;

fig_h = figure(1);
set(gcf,'Position',[110 270 1390 420])
axes1 = axes('Parent',fig_h,'FontSize',fntsize);
hold on;

%background
patch([0 5 5 0],[0 0 1 1],'w','edgecolor',[0 0 0],'edgealpha',1)

% %QoI region
ptsy = [0.625 0.875 0.875 0.625]; %qoi3
ptsx = [0.375 0.375 0.625 0.625]; %qoi3
% ptsy = [2.375 2.625 2.625 2.375]; %qoi6
% ptsx = [0.375 0.375 0.625 0.625]; %qoi6
% ptsy = [0.625 1.5 1.5 0.625]; %qoi7
% ptsx = [0.25 0.25 0.75 0.75]; %qoi7
% ptsy = [0 5 5 0]; %qoi5
% ptsx = [0 0 1 1]; %qoi5
subreg = patch(ptsy, ptsx,...
  [0.87843137254902 0.690196078431373 1], ...
  'EdgeColor','none'); 

%data points
ptsy = [0.35 1.56 3.1];
ptsx = [0.35 0.61 0.5];
% ptsy = [0.35 0.98 1.56 2.46 3.1];
% ptsx = [0.35 0.51 0.61 0.37 0.5];
% ptsy = [0.35 0.98 1.11 1.56 2.23 2.46 2.92 3.1 3.49 4.79];
% ptsx = [0.35 0.51 0.89 0.61 0.51 0.37 0.7 0.5 0.76 0.26];
datapts = scatter(ptsy,ptsx,80, ...
  'MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor',[0 .7 .7]);

leg_h = legend([subreg datapts],{' $\Omega_I$',' Observations'},...
  'Position',[0.776238418335466, 0.412506804633401,...
   0.217572463768116, 0.321180555555556]);
set(leg_h,'FontSize',fntsize)

ylabel('$x_2$','FontSize',fntsize);
xlabel('$x_1$','FontSize',fntsize);
axis equal tight

set(findall(gcf,'type','text'),'FontSize',fntsize)