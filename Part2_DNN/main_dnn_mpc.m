
% Description: 
% Determines DNN approximation to nonlinear MPC problem from paper (*).
% Uses open- and closed-loop training approaches.
% Scenario-based probabilistic validation is also performed. 

% Written by: Joel Paulson
% Date: 06/10/21


%% Initialize script

% clear variables
clear

% import casadi
import casadi.*

% fix random seed for reproducible results 
rng(800,'twister')


%% Specify user inputs for mpc problem

% prediction horizon
N = 10;

% sampling time
Ts = 0.3;

% declare model variables
nx = 2;
nu = 1;
x = SX.sym('x',nx);
u = SX.sym('u',nu);

% fixed model parameters
mu1 = 0.5;

% model equations
dx1 = x(2) + u(1)*(mu1+(1-mu1)*x(1));
dx2 = x(1) + u(1)*(mu1-4*(1-mu1)*x(2));
xdot = vertcat(dx1,dx2);

% stage cost (Lagrange term)
stageCost = x'*[0.5, 0.0 ; 0.0, 0.5]*x + u'*1.0*u;

% terminal cost (Mayer term)
terminalCost = 0;

% time discretization approach
%['DiscreteTime', ForwardEuler', 'RK4', 'Collocation', 'ImplicitEuler']
discretization = 'ForwardEuler';

% sparse or condensed method
%for collocation and implicit Euler we must use sparse approach wherein
%states are introduced as variables at every time step
sparse = 1;

% collocation has extra options
d = 3;
orthogonal_polynomials = 'legendre'; %'radau';

% hard state constraints
x_min = [-inf, -inf]';
x_max = [ inf,  inf]';

% input constraints
u_min = [-2]';
u_max = [ 2]';

% path constraints in standard form g(x,u) <= 0
pathConstraints = vertcat(x(1)-1, -1-x(1), x(2)-1, -1-x(2));

% terminal constraints in standard form gf(x) <= 0
terminalConstraints = SX();

% soft constraint penalty for path and terminal constraints
%use "[]" to turn off, set to scalar > 0 to turn on
%any state bound that you wish to soften should be moved into the path constraint
soft_penalty = 1000;

% initial guess for optimization variables
x_init = [0, 0]';
u_init = [0]';

% solver options
plugin_opts = struct('expand',true,'print_time',0);
solver_opts = struct('max_iter',1000,'tol',1e-8,'print_level',0);


%% Specify user inputs for dnn approximation

% number of nodes and layers in the network
H = 6;
L = 5;

% number of training samples for the open loop version
Ns = 250;

% number of closed-loop simulations and number of steps per simulation
Nsim = 20;
Ncl = ceil(Ns/Nsim);

% bounds for constructing dnn 
x_min = [-1, -1]';
x_max = [ 1,  1]';


%% Define the plant with uncertainty

% plant equations, same as model with additive uncertainty
nw = 2;
w = SX.sym('w', nw);
xdotw = xdot + [w(1) ; w(2)];

% ||w||_1 <= wmax
wmax = 0.2;

% initial simulator state matrix (each row will be run separately)
x0 = [-0.75, -0.5];


%% Select parameters for probabilistic validation

% specify the accuracy level, 1-epsilon
epsilon = 0.05;

% specify the confidence level, 1-delta
delta = 1e-3;


%% Get MPC solver 

% store arguments to be passed to mpc function
mpc_data.N = N;
mpc_data.nx = nx;
mpc_data.nu = nu;
mpc_data.x = x;
mpc_data.u = u;
mpc_data.Ts = Ts;
mpc_data.xdot = xdot;
mpc_data.stageCost = stageCost;
mpc_data.terminalCost = terminalCost;
mpc_data.discretization = discretization;
mpc_data.sparse = sparse;
mpc_data.d = d;
mpc_data.orthogonal_polynomials = orthogonal_polynomials;
mpc_data.x_min = x_min;
mpc_data.x_max = x_max;
mpc_data.u_min = u_min;
mpc_data.u_max = u_max;
mpc_data.pathConstraints = pathConstraints;
mpc_data.terminalConstraints = terminalConstraints;
mpc_data.soft_penalty = soft_penalty;
mpc_data.x_init = x_init;
mpc_data.u_init = u_init;
mpc_data.plugin_opts = plugin_opts;
mpc_data.solver_opts = solver_opts;

% call "get_mpc" to get the opti stack solver that corresponds to the input
% options specified above; all data is stored in the "mpc_data" structure
mpc_data = get_mpc(mpc_data);


%% Open-loop training

% print starting statement
fprintf('running open-loop training procedure...')
tstart_ol = tic;

% initial state values
X0 = (x_max-x_min).*lhsdesign(Ns,nx)' + x_min;

% evaluate mpc for the various samples 
U = zeros(nu,Ns);
for i = 1:Ns
    % specify the initial value
    mpc_data.opti.set_value(mpc_data.X{1}, X0(:,i));
    
    % solve the mpc problem
    try 
        sol = mpc_data.opti.solve();
        Ucl = sol.value(mpc_data.U{1});
    catch
        Ucl = mpc_data.opti.debug.value(mpc_data.U{1});
        Ucl = max(min(Ucl, u_max), u_min);
    end
    
    % store the first optimal input
    U(:,i) = Ucl;
    
    % reset initial condition
    mpc_data.opti.set_initial(mpc_data.Var, mpc_data.Var0);
end

% scale the training data
data = [X0];
data_min = min(data,[],2);
data_max = max(data,[],2);
target = U;
target_min = min(target,[],2);
target_max = max(target,[],2);
data_s = 2*(data-repmat(data_min,[1,Ns]))./repmat(data_max-data_min,[1,Ns])-1;
target_s = 2*(target-repmat(target_min,[1,Ns]))./repmat(target_max-target_min,[1,Ns])-1;

% train the network
net = feedforwardnet(H*ones(1,L), 'trainlm');
for l = 1:L
    net.layers{l}.transferFcn = 'poslin';
end
[net, tr] = train(net, data_s, target_s);

% extract weights and bias
W = cell(L+1,1);
b = cell(L+1,1);
W{1} = net.IW{1};
b{1} = net.b{1};
for i = 1:L
    W{i+1} = net.LW{i+1,i};
    b{i+1} = net.b{i+1};
end

% create casadi function
zs = 2*(x-data_min)./(data_max-data_min)-1;
if strcmp(net.layers{1}.transferFcn, 'poslin')
    zi = max(W{1}*zs+b{1},0);
elseif strcmp(net.layers{1}.transferFcn, 'tansig')
    zi = tansig(W{1}*zs+b{1});
end
for i = 1:L-1
    if strcmp(net.layers{i+1}.transferFcn, 'poslin')
        zi = max(W{i+1}*zi+b{i+1},0);
    elseif strcmp(net.layers{i+1}.transferFcn, 'tansig')
        zi = tansig(W{i+1}*zi+b{i+1});
    end
end
ysout = W{L+1}*zi+b{L+1};
dnn_ol_ca = Function('dnn_ol_ca', {x}, {(ysout+1)/2.*(target_max-target_min)+target_min});
dnn_ol = @(x)(full(dnn_ol_ca(x)));

% print end statement
tend_ol = toc(tstart_ol);
fprintf('done..took %g seconds\n', tend_ol)


%% Closed-loop training

% print starting statement
fprintf('running closed-loop training procedure...')
tstart_cl = tic;

% create integrator object for simulation
dae = struct('x',x,'p',vertcat(u,w),'ode',xdotw,'quad',stageCost); % CVODES from the SUNDIALS suite
opts = struct('tf',Ts);
fsim = integrator('fsim', 'cvodes', dae, opts);

% loop over number of repeats
CL_Trainig_Data = cell(Ncl,1);
for i = 1:Ncl
    % initialize table to store closed-loop simulation data
    Time = (0:Nsim)'*Ts;
    States = zeros(nx,Nsim+1)';
    Inputs = [zeros(nu,Nsim)' ; nan*ones(1,nu)];
    Disturbances = [zeros(nw,Nsim)' ; nan*ones(1,nw)];
    Objective = [zeros(1,Nsim)' ; nan];
    TimeCPU = [zeros(1,Nsim)' ; nan];
    CL_Trainig_Data{i} = table(Time, States, Inputs, Disturbances, Objective, TimeCPU);
    
    % loop over the simulation time
    Xcl = x0;
    Jcl = 0;
    CL_Trainig_Data{i}.States(1,:) = Xcl';
    for k = 1:Nsim
        % solve optimization given the most recent state
        tic
        mpc_data.opti.set_value(mpc_data.X{1}, Xcl)
        try
            sol = mpc_data.opti.solve();
            Ucl = sol.value(mpc_data.U{1});
        catch
            % if optimization fails, take the last iteration
            Ucl = mpc_data.opti.debug.value(mpc_data.U{1});
            Ucl = max(min(Ucl, u_max), u_min);
        end
        CL_Trainig_Data{i}.TimeCPU(k,:) = toc;
        
        % warm start by updating initial condition of optimization with
        % previous solution
        mpc_data.opti.set_initial(sol.value_variables());
        
        % randomly draw most recent disturbances
        Wcl = wmax*(2*rand(nw,1)-1);
        
        % simulate system and update state
        res = fsim('x0', Xcl, 'p', vertcat(Ucl,Wcl));
        Xcl = full(res.xf);
        Jcl = Jcl + (full(res.qf));
        
        % store data in table
        CL_Trainig_Data{i}.States(k+1,:) = Xcl;
        CL_Trainig_Data{i}.Inputs(k,:) = Ucl;
        CL_Trainig_Data{i}.Disturbances(k,:) = Wcl;
        CL_Trainig_Data{i}.Objective(k,:) = Jcl;
    end    
end

% scale the training data
data = [];
target = [];
for i = 1:Ncl
    data = [data ; CL_Trainig_Data{i}.States(1:end-1,:)];
    target = [target ; CL_Trainig_Data{i}.Inputs(1:end-1,:)];
end
data = data';
target = target';
data_min = min(data,[],2);
data_max = max(data,[],2);
target_min = min(target,[],2);
target_max = max(target,[],2);
data_s = 2*(data-repmat(data_min,[1,Ncl*Nsim]))./repmat(data_max-data_min,[1,Ncl*Nsim])-1;
target_s = 2*(target-repmat(target_min,[1,Ncl*Nsim]))./repmat(target_max-target_min,[1,Ncl*Nsim])-1;

% train the network
net = feedforwardnet(H*ones(1,L), 'trainlm');
for l = 1:L
    net.layers{l}.transferFcn = 'poslin';
end
[net, tr] = train(net, data_s, target_s);

% extract weights and bias
W = cell(L+1,1);
b = cell(L+1,1);
W{1} = net.IW{1};
b{1} = net.b{1};
for i = 1:L
    W{i+1} = net.LW{i+1,i};
    b{i+1} = net.b{i+1};
end

% create casadi function
zs = 2*(x-data_min)./(data_max-data_min)-1;
if strcmp(net.layers{1}.transferFcn, 'poslin')
    zi = max(W{1}*zs+b{1},0);
elseif strcmp(net.layers{1}.transferFcn, 'tansig')
    zi = tansig(W{1}*zs+b{1});
end
for i = 1:L-1
    if strcmp(net.layers{i+1}.transferFcn, 'poslin')
        zi = max(W{i+1}*zi+b{i+1},0);
    elseif strcmp(net.layers{i+1}.transferFcn, 'tansig')
        zi = tansig(W{i+1}*zi+b{i+1});
    end
end
ysout = W{L+1}*zi+b{L+1};
dnn_cl_ca = Function('dnn_cl_ca', {x}, {(ysout+1)/2.*(target_max-target_min)+target_min});
dnn_cl = @(x)(full(dnn_cl_ca(x)));

% print end statement
tend_cl = toc(tstart_cl);
fprintf('done..took %g seconds\n', tend_cl)


%% Probabilistic validation of DNN controllers

% print starting statement
fprintf('running validation procedure...')
tstart_val = tic;

% calculate the number of validation samples
Nval = ceil(log(1/delta)/log(1/(1-epsilon)));

% loop over number of validation samples
DNN_Data = cell(Nval,2);
for n = 1:2
    for i = 1:Nval
        % initialize table to store closed-loop simulation data
        Time = (0:Nsim)'*Ts;
        States = zeros(nx,Nsim+1)';
        Inputs = [zeros(nu,Nsim)' ; nan*ones(1,nu)];
        Disturbances = [zeros(nw,Nsim)' ; nan*ones(1,nw)];
        Objective = [zeros(1,Nsim)' ; nan];
        TimeCPU = [zeros(1,Nsim)' ; nan];
        DNN_Data{i,n} = table(Time, States, Inputs, Disturbances, Objective, TimeCPU);
        
        % loop over the simulation time
        Xcl = x0;
        Jcl = 0;
        DNN_Data{i,n}.States(1,:) = Xcl';
        for k = 1:Nsim
            % evaluate the dnn for the most recent state
            tic
            if n == 1
                Ucl = dnn_ol(Xcl);
            elseif n == 2
                Ucl = dnn_cl(Xcl);
            end
            Ucl = max(min(Ucl,u_max),u_min);
            DNN_Data{i,n}.TimeCPU(k,:) = toc;
            
            % warm start by updating initial condition of optimization with
            % previous solution
            mpc_data.opti.set_initial(sol.value_variables());
            
            % randomly draw most recent disturbances
            Wcl = wmax*(2*rand(nw,1)-1);
            
            % simulate system and update state
            res = fsim('x0', Xcl, 'p', vertcat(Ucl,Wcl));
            Xcl = full(res.xf);
            Jcl = Jcl + (full(res.qf));
            
            % store data in table
            DNN_Data{i,n}.States(k+1,:) = Xcl;
            DNN_Data{i,n}.Inputs(k,:) = Ucl;
            DNN_Data{i,n}.Disturbances(k,:) = Wcl;
            DNN_Data{i,n}.Objective(k,:) = Jcl;
        end
    end
end

% test if state constraints were violated for open-loop dnn
dnn_ol_states_over_all_validation_runs = [];
for i = 1:Nval
    dnn_ol_states_over_all_validation_runs = [dnn_ol_states_over_all_validation_runs ; DNN_Data{i,1}.States];
end
ol_wc_upper = max(max((dnn_ol_states_over_all_validation_runs - x_max'))); % calcualte the worst-case distance from the upper bound (should be negative if satisifed)
ol_wc_lower = max(max(-(dnn_ol_states_over_all_validation_runs - x_min'))); % calcualte the negative of the worst-case distance from the lower bound (should be negative if satisifed)
ol_wc = max(ol_wc_upper, ol_wc_lower);
if ol_wc <= 0
    phi_test_ol = 0;
    fprintf('\n\n     test PASSED for dnn_ol!')
else
    phi_test_ol = 1;
    fprintf('\n\n     test FAILED for dnn_ol!')    
end

% test if state constraints were violated for open-loop dnn
dnn_cl_states_over_all_validation_runs = [];
for i = 1:Nval
    dnn_cl_states_over_all_validation_runs = [dnn_cl_states_over_all_validation_runs ; DNN_Data{i,2}.States];
end
cl_wc_upper = max(max((dnn_cl_states_over_all_validation_runs - x_max'))); % calcualte the worst-case distance from the upper bound (should be negative if satisifed)
cl_wc_lower = max(max(-(dnn_cl_states_over_all_validation_runs - x_min'))); % calcualte the negative of the worst-case distance from the lower bound (should be negative if satisifed)
cl_wc = max(cl_wc_upper, cl_wc_lower);
if cl_wc <= 0
    phi_test_cl = 0;
    fprintf('\n\n     test PASSED for dnn_cl!')
else
    phi_test_cl = 1;
    fprintf('\n\n     test FAILED for dnn_cl!')    
end

% plot the states for visualization purposes
for n = 1:2
    figure; hold on;
    for k = 1:nx
        subplot(nx,1,k); hold on;
        if k == 1
            if n == 1
                title('validation of open-loop DNN-MPC')
            elseif n == 2
                title('validation of closed-loop DNN-MPC')
            end            
        end
        for i = 1:Nval
            plot(DNN_Data{i,n}.Time, DNN_Data{i,n}.States(:,k), '-b', 'linewidth', 2)
        end
        set(gcf,'color','w');
        set(gca,'FontSize',20)
        ylabel(['x' num2str(k)])
        if k == 1 || k == 2
            plot([DNN_Data{1,n}.Time(1), DNN_Data{1,n}.Time(end)], [0, 0], ':k', 'linewidth', 2)
            plot([DNN_Data{1,n}.Time(1), DNN_Data{1,n}.Time(end)], [1, 1], '--r', 'linewidth', 2)
            plot([DNN_Data{1,n}.Time(1), DNN_Data{1,n}.Time(end)], [-1, -1], '--r', 'linewidth', 2)
        end
    end
    xlabel('time')
end

% print end statement
tend_val = toc(tstart_val);
fprintf('\n\ndone...took %g seconds\n', tend_val)
