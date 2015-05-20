%easier way to do refinement...

refPortion = 0.12;

errFileID = ...
  fopen('error_est_breakdown.dat','r');
errFormatSpec = '%f %f %f';
errSizeMat = [3 Inf];

A = fscanf(errFileID,errFormatSpec,errSizeMat);
A = A';

divFileID = ...
  fopen('divvy.txt','r');
divFormatSpec = '%d %d';
divSizeMat = [2 Inf];

%if starting from LF
% divvy = zeros(size(A,1),1);

% %if starting from MF
divvy = fscanf(divFileID,divFormatSpec,divSizeMat);
divvy = divvy';
divvy = divvy(:,2);

nElem = length(divvy);
cutoff = round((1-refPortion)*nElem);
err = A(:,3);
meep = sort(err);
bloop = (err >= meep(cutoff));
bloop = bloop + 0;
divvynew = bloop + divvy;
divvynew(divvynew == 2) = 1;

figure(1)
scatter(A(:,1),A(:,2),200,divvynew,'s','filled','MarkerEdgeColor','k')
set(gcf,'Position',[278 395 1323 183])

figure(2)
scatter(A(:,1),A(:,2),200,A(:,3),'s','filled')
set(gcf,'Position',[278 395 1323 183])


divvynew = [(0:1:(nElem-1))' divvynew];
dlmwrite('do_divvy.txt',divvynew,' ')