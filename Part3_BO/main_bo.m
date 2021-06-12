
% Description: 
% Solves MPC tuning problem using constrained Bayesian optimization.
% User specifications are given at the beginning of the script. 

% Written by: Joel Paulson
% Date: 06/09/21

% clear variables
clear

% fix random seed for repeatable results
rng(100,'twister')

% maximum number of objective calls
Nmax = 10;

% number of times to call bayesopt to get distribution of "optimal" points
Nrepeat = 5;

% number of simulation steps
T = 40;

 % number of Monte carlo samples
M = 1;

% define bayesian optimization variables
xbo1 = optimizableVariable('backoff1',[0 0.5],'Type','real');
xbo2 = optimizableVariable('backoff2',[0 0.5],'Type','real');
xbo3 = optimizableVariable('Npred',[5 10],'Type','integer');
xbo4 = optimizableVariable('discretization',{'ForwardEuler', 'RK4', 'Collocation', 'ImplicitEuler'},'Type','categorical');
xbo = [xbo1 ; xbo2 ; xbo3 ; xbo4];

% objective function and constraint handle
plot_on = 0; % turn plotting off
fun = @(x)run_closed_loop_system(x, T, M, plot_on);

% loop over number of runs
saved_results = cell(Nrepeat,1);
for i = 1:Nrepeat
    
    % print statement
    startTime_i = tic;
    fprintf('\n****************************************************\n')
    fprintf('running closed-loop simulation %g of %g...\n', i, Nrepeat)
    fprintf('****************************************************\n\n')
        
    % call Bayesian optimization solver
    results = bayesopt(fun,xbo,...
        'AcquisitionFunctionName', 'expected-improvement',...
        'IsObjectiveDeterministic', 0,...
        'ExplorationRatio', 0.5,...
        'GPActiveSetSize', 300,...
        'UseParallel', false,...
        'MaxObjectiveEvaluations', Nmax,...
        'AreCoupledConstraintsDeterministic', [false, false],...
        'NumCoupledConstraints', 2, ...
        'PlotFcn', [], ...
        'InitialX', [], ...
        'InitialObjective', [], ...
        'InitialConstraintViolations', [], ...
        'NumSeedPoints', length(xbo)+1);
    saved_results{i} = results;

    % create table or add to table
    if i == 1
        x_opt = results.XAtMinEstimatedObjective;
        f_opt = results.MinEstimatedObjective;
    else
        x_opt(end+1,:) = results.XAtMinEstimatedObjective;
        f_opt(end+1,:) = results.MinEstimatedObjective;
    end
    
    % print end statement
    endTime_i = toc(startTime_i);
    fprintf('\n TIME REPORT: simulation %g of %g took %g seconds \n\n', i, Nrepeat, endTime_i)    
end

% plot the average minimium function values and error bars
figure; hold on;
feasTrace = zeros(Nrepeat,Nmax);
objMinTrace = zeros(Nrepeat,Nmax);
for i = 1:Nrepeat
    for j = 1:Nmax
        feasTrace(i,j) = saved_results{i}.FeasibilityTrace(j);
        objMinTrace(i,j) = min(saved_results{i}.ObjectiveTrace(1:j));
    end
end
if Nrepeat > 1
    stairs(1:Nmax, mean(objMinTrace), '-b', 'linewidth', 3);
    errorbar(1:Nmax, mean(objMinTrace), 1.96/sqrt(Nrepeat)*std(objMinTrace), '-b', 'CapSize', 10, 'LineStyle', 'none')
else
    stairs(1:Nmax, mean(objMinTrace), '-b', 'linewidth', 3);
end
set(gcf,'color','w')
set(gca,'FontSize',20)
xlabel('number of iterations')
ylabel('negative moles of product C')

% find the index of the median solution found at the final iteration
index = find(f_opt == median(f_opt));

% run the closed-loop system at the (median) optimal tuning values and plot
% the results
run_closed_loop_system(x_opt(index,:), T, 5, 1);
