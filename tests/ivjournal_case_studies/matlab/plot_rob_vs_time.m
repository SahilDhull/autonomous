T1= T;
if opt.spec_space == 'X'
    YT1 = XT;
else
    YT1 = YT;
end
simTime = T1(end);
psi = phi;

bs = zeros(length(T1),2); % Boolean signals for each atomic proposition i
for i = 1:length(preds)
    bs(:,i) = YT1*preds(i).A'<=preds(i).b;
end

% The following is from a reactive requirement: psi = '[](p1 -> <>_[0,3] ( p2 /\ <>_[0,3] ( p1 /\ <>_[0,4] p2 ) ))';

rob_psi = zeros(length(T1),1); % store the robustness values of psi
fals_psi = zeros(length(T1),1); % store the robustness values of psi
aux_psi = cell(0);
for ii = 1:length(T1)
    %[rob_psi(ii), aux_psi{ii}] = dp_taliro(psi,preds,YT1(ii:end,:),T1(ii:end));
    [rob_psi(ii), aux_psi{ii}] = dp_taliro(psi,preds,YT1(1:ii,:),T1(1:ii));
end
fals_psi = rob_psi > 0;

% Plot predicates:
figure
i = 1;
for i = 1:length(preds)
    subplot(length(preds),1,i)
    plot(T1, bs(:,i));
    title(strrep(preds(i).str,'_','\_'))
    grid on
    axis([0 simTime -0.5 1.5])
end

% Plot falsification:
figure
title('Falsification')
plot(T1,fals_psi,'b')
hold on;
plot(T1,rob_psi,'r')
grid on
%axis([0 simTime -0.5 1.5])
