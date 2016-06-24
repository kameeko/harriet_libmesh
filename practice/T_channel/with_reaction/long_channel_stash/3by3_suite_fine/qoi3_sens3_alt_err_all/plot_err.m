%plot error vs refinement

close all;

fntsize = 13;
set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');

errEst = [-0.0018293936; -9.84E-005; 1.41E-005];
errTrue = [-0.0016895481; -0.0001003636; 1.32748289514697E-005];
qoi = [0.0050801785; 0.0034909939; 0.0033773555];
% relErrEst = abs(errEst./qoi);
% relErrTrue = abs(errTrue./qoi);
relErrEst = abs(errEst./(qoi+errEst)); %relative to predicted I_HF
qoiHF = 0.0033906304;
relErrTrue = abs(errTrue/qoiHF);
ref = [0; 0.0578; 0.105];

figure('Position',[470 580 765 205])
plot(100*[ref; 1], [relErrTrue; 0], '-*', 100*[ref; 1], [relErrEst; 0], '-*','LineWidth',1.5);
legend('True','Estimated')
title('True and Estimated Absolute Relative Error in QoI','FontWeight','normal')
ylabel({'Absolute Relative', '\quad Error in QoI'})
xlabel('$\%$ HF')
set(gca,'FontSize',fntsize); 
