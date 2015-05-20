%to create "setup" plots
close all;

ftsize = 13;

fig_h = figure(1);
% set(gcf,'Position',[1 1 1150 240])
% set(gcf,'Position',[1 1 1150 300])
set(gcf,'Position',[1 1 300/2.1 1150/4])
axes1 = axes('Parent',fig_h,'FontSize',ftsize);
hold on;

%background
% patch([0 5 5 0],[0 0 1 1],'w')
patch([0 0 1 1], [0 5 5 0],'w')

% %QoI region
ptsx = [0.625 0.875 0.875 0.625]; %qoi3
ptsy = [0.375 0.375 0.625 0.625]; %qoi3
% ptsx = [2.375 2.625 2.625 2.375]; %qoi6
% ptsy = [0.375 0.375 0.625 0.625]; %qoi6
% ptsx = [0.625 1.5 1.5 0.625]; %qoi7
% ptsy = [0.25 0.25 0.75 0.75]; %qoi7
% ptsx = [0 5 5 0]; %qoi5
% ptsy = [0 0 1 1]; %qoi5
subreg = patch(ptsy, ptsx,...
  [0.87843137254902 0.690196078431373 1], ...
  'EdgeColor','none'); 

%data points
% ptsx = [0.35 1.56 3.1];
% ptsy = [0.35 0.61 0.5];
% ptsx = [0.35 0.98 1.56 2.46 3.1];
% ptsy = [0.35 0.51 0.61 0.37 0.5];
ptsx = [0.35 0.98 1.11 1.56 2.23 2.46 2.92 3.1 3.49 4.79];
ptsy = [0.35 0.51 0.89 0.61 0.51 0.37 0.7 0.5 0.76 0.26];
datapts = scatter(ptsy,ptsx,30, ...
  'MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor',[0 .7 .7]);

% leg_h = legend([subreg datapts],{'\Omega_I','Observations'},...
%   'Position',[0.776238418335466, 0.412506804633401,...
%    0.217572463768116, 0.321180555555556]);
% set(leg_h,'FontSize',20)

ylabel('x','FontSize',ftsize);
xlabel('{ }','FontSize',ftsize);
% title('Location of Observations and QoI Region','FontSize',ftsize)
set(gca,'xtickLabel',{' ',' ',' '})
set(gca,'XDir','Reverse')

set(findall(gcf,'type','text'),'FontSize',ftsize)