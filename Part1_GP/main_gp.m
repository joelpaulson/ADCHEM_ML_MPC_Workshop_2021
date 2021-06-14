
% Description: 
% Uses a Gaussian process to learn plant-model mismatch.
% Linear model identification is first performed to build a nominal model. 
% User specifications are given at the beginning of the script. 

% Written by: Joel Paulson and Ketong Shao (edited by Ali Mesbah)
% Date: 06/11/21

% clear variables
clear

% fix random seed for reproducible results 
rng(250,'twister')

% number of repeat simulations
Nrepeat = 2;

% number of simulation steps
N = 15;

% input range
u_min = -1;
u_max = 1;

% noise standard deviation
wstd = 1e-2;

% initial condition
x0 = [-9 ; -4.5];

% kernel function to use in the gaussian process (GP)
kernelFunction = 'ardsquaredexponential';
% MANY ARE AVAILABLE INCLUDING:
% 'exponential'             Exponential kernel.
% 'squaredexponential'      Squared exponential kernel.
% 'matern32'                Matern kernel with parameter 3/2.
% 'matern52'                Matern kernel with parameter 5/2.
% 'rationalquadratic'       Rational quadratic kernel.
% 'ardexponential'          Exponential kernel with a separate length scale per predictor.
% 'ardsquaredexponential'	Squared exponential kernel with a separate length scale per predictor.
% 'ardmatern32'             Matern kernel with parameter 3/2 and a separate length scale per predictor.
% 'ardmatern52'             Matern kernel with parameter 5/2 and a separate length scale per predictor.
% 'ardrationalquadratic'

% input value to hold constant when doing GP predictions wrt (x1,x2)
u_constant = 0;

% values to hold constant when doing GP predictions wrt x1
x2_constant = [0, -2, -4];

% get plant data under random input and noise values
X = zeros(2,N+1,Nrepeat);
U = zeros(1,N,Nrepeat);
W = zeros(1,N,Nrepeat);
for j = 1:Nrepeat % perform Nrepeat simulations
    % specify the same initial condition for each simulation
    X(:,1,j) = x0;
    
    for i = 1:N % each simulation is length N
        % get random input
        U(1,i,j) = (u_max-u_min)*rand+u_min;
        
        % get random noise realization
        W(1,i,j) = wstd*randn;
        
        % give values to plant
        X(:,i+1,j) = plant(X(:,i,j),U(:,i,j),W(:,i,j));
    end
end

% create multi-experiment identification data set (X and U are measured)
id_data_list = cell(Nrepeat,1);
for j = 1:Nrepeat
    Yj = X(:,1:end-1,j)';
    Uj = U(:,:,j)';
    id_data_list{j} = iddata(Yj, Uj);
end

% merge identification data into single iddata object
id_data_merged = id_data_list{1};
for j = 2:Nrepeat
    id_data_merged = merge(id_data_merged, id_data_list{j});
end

% run linear system identification with nx=2 states
nx = 2;
sys = n4sid(id_data_merged,nx);

% extract state-space matrices and convert to standard form 
A = sys.C*sys.A*inv(sys.C);
B = sys.C*sys.B;
Bd = [1 ; 0]; % here we assume we know the subspace that the disturbance lives in

% run this command to see how well the system identification did 
% (there will be decent errors due to the plant-model mismatch)
%compare(id_data_merged, sys)

% evaluate the one-step prediction error for all the identification data
Bd_dagger = pinv(Bd);
XUdata = [];
Wdata = [];
for j = 1:Nrepeat
    for i = 1:N
        XUdata = [XUdata ; X(:,i,j)', U(:,i,j)'];
        Wdata = [Wdata ; (Bd_dagger*(X(:,i+1,j) - ( A*X(:,i,j) + B*U(:,i,j) )))'];
    end
end

% use the prediction error data to train a GP model
gp = fitrgp(XUdata,Wdata,'KernelFunction',kernelFunction,'Standardize',0);

% get the min/max values from the training data
x_min = min(XUdata(:,1:2));
x_max = max(XUdata(:,1:2));

% specify the grids for the first (x1) and second (x2) state for plotting
x1_draw = linspace(x_min(1), x_max(1), 101);
x2_draw = linspace(x_min(2), x_max(2), 101);
[X1_draw, X2_draw] = meshgrid(x1_draw, x2_draw);

% predict the GP model (mean + std) at the grid points
[muW_pred, stdW_pred] = predict(gp,[X1_draw(:), X2_draw(:), u_constant*ones(length(x1_draw)*length(x2_draw),1)]);
muW_pred = reshape(muW_pred, [length(x2_draw), length(x1_draw)]);
stdW_pred = reshape(stdW_pred, [length(x2_draw), length(x1_draw)]);

% plot the mean GP function and the training data
figure; hold on
title(['GP mean, kernel = ' kernelFunction])
h = surf(X1_draw, X2_draw, muW_pred);
scatter3(XUdata(:,1), XUdata(:,2), Wdata, 500, '.r')
set(h, 'EdgeColor', 'none');
view(3)
set(gcf,'color','w');
set(gca,'FontSize',16)
xlabel('x1')
ylabel('x2')
zlabel('w mean')
colorbar

% plot the standard deviation in the GP predictions
% (cyan lines represent 1d traces that are plotted separately)
figure; hold on
title(['GP std, kernel = ' kernelFunction])
h = pcolor(X1_draw, X2_draw, stdW_pred);
scatter(XUdata(:,1), XUdata(:,2), 500, '.r')
for i = 1:length(x2_constant)
    plot(x1_draw, x2_constant(i)*ones(length(x1_draw),1), '-c', 'linewidth', 2)
end
set(h, 'EdgeColor', 'none');
set(gcf,'color','w');
set(gca,'FontSize',16)
xlabel('x1')
ylabel('x2')
colorbar

% loop over the number of x2_constant values
for i = 1:length(x2_constant)
    % calculate the GP predictions in x1 for constant x2 value
    [muw_pred, stdw_pred] = predict(gp,[x1_draw', x2_constant(i)*ones(length(x1_draw),1), u_constant*ones(length(x1_draw),1)]);
    x1_conf = [x1_draw, x1_draw(end:-1:1)];
    w_conf = [(muw_pred+3*stdw_pred)' (muw_pred(end:-1:1)-3*stdw_pred(end:-1:1))'];
    
    % plot the mean and 99% confidence region
    figure; hold on;
    title(['x2 constant = ' num2str(x2_constant(i))])
    plot(x1_draw',muw_pred,'linewidth',2,'color','b')
    h1 = fill(x1_conf,w_conf,'red','EdgeColor','None');
    h1.FaceColor = [0,0.475,0.698];
    h1.FaceAlpha = 0.3;
    set(gcf,'color','w');
    set(gca,'FontSize',16)
    xlabel('x1')
    ylabel('w')
    legend('mean GP prediction', '99% confidence region');
end
