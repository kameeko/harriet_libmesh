%plot error vs refinement

close all;

fntsize = 13;
set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');

errEst = [0.0022021921; -7.03E-004; 3.57E-003; 5.81E-004; 8.30E-004; 3.92E-004; 1.69E-005];
errTrue = [0.0018260025; 0.0006062894; 0.0008967747; 0.00060295; 0.0008887619; 0.0004020916; 1.69478960580405E-005];
qoi = [0.0018016233; 0.0030213363; 0.002730851; 0.0030246758; 0.0027388639; 0.0032255341; 0.0036106779];
relErrEst = abs(errEst./qoi);
relErrTrue = abs(errTrue./qoi);
ref = [0; 0.1168; 0.206; 0.3032; 0.4049; 0.5006; 0.6023];

figure('Position',[470 580 765 205])
plot(100*[ref; 1], [relErrTrue; 0], '-*', 100*[ref; 1], [relErrEst; 0], '-*','LineWidth',1.5);
legend('True','Estimated')
title('True and Estimated Absolute Relative Error in QoI','FontWeight','normal')
ylabel({'Absolute Relative', '\quad Error in QoI'})
xlabel('$\%$ HF')
set(gca,'FontSize',fntsize); 
