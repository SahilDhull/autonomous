close all
%prepare_data_for_plot_scenario3_r5

set(0,'defaultAxesFontSize',36)

% Plot predicates:
figure
i = 1;
subplot(length(preds)+1,1,i)
title('Evaluation of Future Collision Estimation')
%xlabel('Simulation time')
yyaxis left
stairs(T1, bs(:,i), 'LineWidth', 2, 'Color', 'k');
ylabel({'Collision', 'Estimated (C)'}, 'Color', 'k')
grid on
tempcriticaltimes = sort([criticaltimes, rising_criticaltimes]);
axis([5.25 6.15 -0.1 5.0])
yticks([0 1])
yticklabels({'False','True'})

yyaxis right
hold on
plot(T1, YT1(:, AGENT_1_FUTURE_MIN_DISTANCE), 'LineWidth', 1, 'Color', 'r')
hold on
ylabel('Min. future dist.')
hold on
plot([T1(1), T1(end)], [0.5, 0.5], 'r--')
ax = gca;
ax.YAxis(1).Color = 'k';
axis([5.25 6.15 -2.3 8.0])
yticks([0.5 4 8])
%xticks([round(max(0, min(tempcriticaltimes) - 0.1), 1), tempcriticaltimes, round(min(simTime, max(tempcriticaltimes) + 0.1), 1)])

i = 2;
subplot(length(preds)+1,1,i)
title('Evaluation of Applied Brake')
%xlabel('Simulation time')
yyaxis left
stairs(T1, bs(:,i), 'LineWidth', 2, 'Color', 'k');
ylabel({'Excessive', 'braking (B)'}, 'Color', 'k')
grid on
axis([5.25 6.15 -0.1 5.0])
yticks([0 1])
yticklabels({'False','True'})
drawnow
% for jj = 1:1
%     Annotate(gca, 'doublearrow', [rising_criticaltimes(jj),rising_criticaltimes(jj)+0.6], [-0.2*jj,-0.2*jj], 'linestyle', '--', 'color', 'k');
%     %Annotate(gca, 'textbox', [rising_criticaltimes(jj)+0.25, rising_criticaltimes(jj)+0.35], [-0.4,-0.2], 'backgroundcolor', 'none', 'EdgeColor', 'none', 'string', 't1', 'color', 'k'); 
% end

yyaxis right
hold on
plot(T1, max(0.0, -YT1(:, EGO_THROTTLE)), 'LineWidth', 1, 'Color', 'r')
ylabel('Brake (br)')
hold on
plot([T1(1), T1(end)], [0.5, 0.5], 'r--')
axis([5.25 6.15 -0.5 1.2])
yticks([0 0.5 1])
yticklabels({'0.0','0.5','1.0'})
xticks([round(max(0, min(tempcriticaltimes) - 0.1), 1), [tempcriticaltimes(1:2), tempcriticaltimes(4:end)], round(min(simTime, max(tempcriticaltimes) + 0.1), 1)])
ax = gca;
ax.YAxis(1).Color = 'k';



subplot(3,1,3)
plot(T1, brake_next_not_brake, 'Color', 'k');
axis([5.25 6.15 -0.5 1.5+(length(brake_next_not_brake_events)-1)*0.2])
drawnow
colorOrder = get(gca, 'ColorOrder');
title('B \wedge O\neg B (edge)')
xlabel('Simulation time')
color_array = {'b', 'r', 'm', 'c', 'g'};
for i = 1:length(brake_next_not_brake_events)
    hold on
    timestartval = T1(brake_next_not_brake_events(i));
    timeendval = T1(brake_next_not_brake_events(i))+0.5;
    nextcolor = color_array{mod(i, length(color_array))}; %colorOrder(mod(length(get(gca, 'Children')), size(colorOrder, 1))+1, :);
    plot(timestartval, 1, '*', 'Color', nextcolor, 'MarkerSize',10)
    Annotate(gca, 'textbox', [timestartval-0.04,timestartval-0.01], [0.95,1.25], 'backgroundcolor', 'none', 'EdgeColor', 'none', 'string', ['(', num2str(i), ')'], 'color', nextcolor, 'fontsize', 30);
    if i<3
        Annotate(gca, 'doublearrow', [timestartval,timeendval], [1.2 + 0.3*(i-1),1.2 + 0.3*(i-1)], 'linestyle', '--', 'color', nextcolor);
        Annotate(gca, 'textbox', [((timestartval+timeendval)/2)-0.01, ((timestartval+timeendval)/2)+0.01], [1.2 + 0.3*(i-1)+0.01,1.2+ 0.3*(i-1)+0.3], 'backgroundcolor', 'none', 'EdgeColor', 'none', 'string', 't2', 'color', nextcolor, 'fontsize', 30); 
    end
end
%grid on
ax = gca;
ax.XGrid = 'on';
ax.GridLineStyle = '-';
xticks([round(max(0, min(criticaltimes) - 0.1), 1), criticaltimes, round(min(simTime, max(criticaltimes) + 0.1), 1)])
yticks([0 1])
yticklabels({'False','True'})


% % Plot falsification:
% figure
% title('Falsification')
% plot(T1,fals_psi,'b')
% hold on;
% plot(T1,rob_psi,'r')
% grid on
% %axis([0 simTime -0.5 1.5])
