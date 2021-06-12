function [f,c] = run_closed_loop_system(xbo, Nsim, M, plot_on, varargin)

% Description:
% This function runs closed-loop mpc for a given set of tuning parameters
% for the semibatch reactor

% Written by: Joel Paulson
% Date: 11/10/20

% import casadi
import casadi.*


%% Specify user inputs

% parse inputs
if length(varargin) == 1
    save_cl = varargin{1};
elseif length(varargin) == 2
    save_cl = varargin{1};
    rng(varargin{2},'twister') % fix random seed
else
    save_cl = [];
end

% convert input into set of tuning parameters
backoff = [xbo.backoff1 ; xbo.backoff2]; % backoff for path constraints upper and lower bound on Temp
N = xbo.Npred;
discretization = xbo.discretization;

% % prediction horizon
% N = 10;

% sampling time
Ts = 30;

% declare model variables
nx = 7;
nu = 2;
x = SX.sym('x', nx);
u = SX.sym('u', nu);

% fixed model parameters
dH = -355;
k0 = 3.3470e-7;
rho = 1000;
cp = 4.2;
r = 0.092;
VJ = 2.22e-3;
VdotJin = 9.167e-5;
alpha = 0.14844;
tauc = 900;
cBin = 3000;
Tin = 300;

% model equations
Vr = x(1);
cA = x(2);
cB = x(3);
cC = x(4);
Tr = x(5);
TJ = x(6);
TJin = x(7);
Vdotin = u(1);
TJinset = u(2);
Aw = 2*Vr/r + pi*r^2;
dx1 = Vdotin;
dx2 = -Vdotin/Vr*cA - k0*cA*cB;
dx3 = Vdotin/Vr*(cBin-cB) - k0*cA*cB;
dx4 = -Vdotin/Vr*cC + k0*cA*cB;
dx5 = Vdotin/Vr*(Tin-Tr) - alpha*Aw*(Tr-TJ)/(rho*Vr*cp) - k0*cA*cB*dH/(rho*cp);
dx6 = VdotJin/VJ*(TJin-TJ) + alpha*Aw*(Tr-TJ)/(rho*VJ*cp);
dx7 = (TJinset-TJin)/tauc;
xdot = vertcat(dx1, dx2, dx3, dx4, dx5, dx6, dx7);

% stage cost (Lagrange term)
stageCost = -Vr*cC;

% terminal cost (Mayer term)
terminalCost = 0;

% % time discretization approach
% %['DiscreteTime', ForwardEuler', 'RK4', 'Collocation', 'ImplicitEuler']
% discretization = 'ForwardEuler';

% sparse or condensed method
%for collocation and implicit Euler we must use sparse approach wherein
%states are introduced as variables at every time step
sparse = 1;

% collocation has extra options
d = 3;
orthogonal_polynomials = 'legendre'; %'radau';

% state constraints
x_min = [-inf, -inf, -inf, -inf, -inf, -inf, -inf]';
x_max = [ inf,  inf,  inf,  inf,  inf,  inf,  inf]';

% input constraints
u_min = [0, 280]';
u_max = [9e-6, 350]';

% path constraints in standard form g(x,u) <= 0
pathConstraints = vertcat(x(5)-326, 322-x(5));

% terminal constraints in standard form gf(x) <= 0
terminalConstraints = SX();

% soft constraint penalty for path and terminal constraints
%use "[]" to turn off, set to scalar > 0 to turn on
%any state bound that you wish to soften should be moved into the path constraint
soft_penalty = 1000;

% % backoff values for path constraints
% backoff = [0.5 ; 0.05]*0;

% initial guess for optimization variables
x_init = [3.5e-3, 2000, 0, 0, 325, 325, 325]';
u_init = [3e-6, 300]';

% solver options
plugin_opts = struct('expand',true,'print_time',0);
solver_opts = struct('max_iter',1000,'tol',1e-8,'print_level',0);

% % number of plant simulation steps
% Nsim = 40;

% number of plant repeat simulations
Nrepeat = M;

% plant equations
nw = 3;
w = SX.sym('w', nw);
xdotw = xdot + [0;0;w(1);0;0;w(2);w(3)];

% initial simulator state
x0 = [3.5e-3 ; 2000 ; 0 ; 0 ; 325 ; 325 ; 325];

% noise standard deviation
wrange = [0.5 ; 0.05 ; 0.05];


%% Create MPC problem

% create opti instance (where we will build and store solver)
opti = casadi.Opti();

% setup dynamic functions based on different cases
switch discretization    
    % discrete time dynamics already specified
    case {'DiscreteTime'}
        f = Function('f', {x, u}, {xdot});
        L = Function('L', {x, u}, {stageCost});
        
    % need basic function for forward Euler
    case {'ForwardEuler'}
        f = Function('f', {x, u}, {x + Ts*xdot});
        L = Function('L', {x, u}, {stageCost*Ts});
        
    % fixed step Runge-Kutta 4 integrator    
    case {'RK4'}
        M = 4; % RK4 steps per interval
        DT = Ts/M;
        fcont = Function('fcont', {x, u}, {xdot, stageCost});
        X0 = MX.sym('X0', nx);
        U = MX.sym('U', nu);
        X = X0;
        Q = 0;
        for j=1:M
            [k1, k1_q] = fcont(X, U);
            [k2, k2_q] = fcont(X + DT/2 * k1, U);
            [k3, k3_q] = fcont(X + DT/2 * k2, U);
            [k4, k4_q] = fcont(X + DT * k3, U);
            X = X+DT/6*(k1 +2*k2 +2*k3 +k4);
            Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
        end
        f = Function('f', {X0, U}, {X});
        L = Function('L', {X0, U}, {Q});

    % get continuous time function for implicit Euler
    case {'ImplicitEuler'}
        fcont = Function('fcont', {x, u}, {xdot});
        L = Function('L', {x, u}, {stageCost*Ts});        
        
    % get the necessary expressions to implement collocation equations
    case {'Collocation'}
        [B, C, D] = GetCollocationCoefficients(d, orthogonal_polynomials);
        fcont = Function('fcont', {x, u}, {xdot});
        Lcont = Function('Lcont', {x, u}, {stageCost});
        
    % error thrown if other specification is included
    otherwise
        error('Discretization type not supported!')        
end

% terminal cost function
Mterm = Function('Mterm', {x}, {terminalCost});

% path constraints
npc = length(pathConstraints);
g = Function('g', {x, u}, {pathConstraints});

% terminal constraints
ntc = length(terminalConstraints);
gf = Function('gf', {x}, {terminalConstraints});

% initialize cell for all states and inputs over horizon
X = cell(N+1,1);
U = cell(N,1);
if strcmpi(discretization,'Collocation') % add additional states at collocation points
    Xc = cell(N,d);
end
if ~isempty(soft_penalty) % add penalty terms if needed
    E = cell(N,1);
    if strcmpi(discretization,'Collocation') % add additional softening at collocation points
        Ec = cell(N,d);
    end
end

% define parameters of the optimal control problem
X{1} = opti.parameter(nx);

% loop over time to construct optimal control problem
J = 0;
for k = 1:N
    % new variable for the control input
    U{k} = opti.variable(nu);
    opti.subject_to( u_min <= U{k} <= u_max )
    opti.set_initial(U{k}, u_init)

    % add path constraints
    if npc > 0
        if isempty(soft_penalty)
            opti.subject_to( g(X{k},U{k}) + backoff <= zeros(npc,1) )            
        else
            E{k} = opti.variable(npc);
            opti.subject_to( E{k} >= zeros(npc,1) )
            opti.set_initial(E{k}, zeros(npc,1))
            opti.subject_to( g(X{k},U{k}) + backoff <= E{k} )
            J = J + soft_penalty*(E{k}'*E{k});
        end
    end
    
    % handle different cases
    switch discretization
        % same basic structure for forward discretization schemes
        case {'DiscreteTime', 'ForwardEuler', 'RK4'}
            % evaluate dynamics and contribution to objective
            Xk_end = f(X{k}, U{k});
            J = J + L(X{k}, U{k});
            
            % either add new states or set equal to previous evaluation
            switch sparse
                case true
                    X{k+1} = opti.variable(nx);
                    opti.set_initial(X{k+1}, x_init)
                    opti.subject_to( X{k+1} == Xk_end )
                otherwise
                    X{k+1} = Xk_end;
            end
            
            % add state constraints
            opti.subject_to( x_min <= X{k+1} <= x_max )
                                
        % implicit scheme has different constraints
        case {'ImplicitEuler'}
            % add state variables for the end of time step
            X{k+1} = opti.variable(nx);
            opti.set_initial(X{k+1}, x_init)
            
            % impose implicit dynamic scheme
            opti.subject_to( X{k+1} == X{k} + Ts*fcont(X{k+1},U{k}) )
            
            % add contribution to objective
            J = J + L(X{k}, U{k});

            % add state constraints
            opti.subject_to( x_min <= X{k+1} <= x_max )            
            
        % collocation equations for state dynamics and objective
        case {'Collocation'}
            % add state variables at collocation points
            for j = 1:d
                Xc{k,j} = opti.variable(nx);
                opti.set_initial(Xc{k,j}, x_init)
                opti.subject_to( x_min <= Xc{k,j} <= x_max )
                if npc > 0
                    if isempty(soft_penalty)
                        opti.subject_to( g(Xc{k,j},U{k}) + backoff <= zeros(npc,1) )
                    else
                        Ec{k,j} = opti.variable(npc);
                        opti.subject_to( Ec{k,j} >= zeros(npc,1) )
                        opti.set_initial(Ec{k,j}, zeros(npc,1))
                        opti.subject_to( g(Xc{k,j},U{k}) + backoff <= Ec{k,j} )
                        J = J + soft_penalty*(Ec{k,j}'*Ec{k,j});                        
                    end
                end
            end
            
            % loop over collocation points
            Xk_end = D(1)*X{k};
            for j = 1:d
                % expression for the state derivative at the collocation point
                xp = C(1,j+1)*X{k};
                for r = 1:d
                    xp = xp + C(r+1,j+1)*Xc{k,r};
                end
                
                % append collocation equations
                opti.subject_to( xp == Ts*fcont(Xc{k,j},U{k}) )
                
                % add contribution to the end state
                Xk_end = Xk_end + D(j+1)*Xc{k,j};

                % add contribution to quadrature function
                J = J + B(j+1)*Lcont(Xc{k,j},U{k})*Ts;
            end
            
            % add state variables for the end of time step
            X{k+1} = opti.variable(nx);
            opti.set_initial(X{k+1}, x_init)
            opti.subject_to( x_min <= X{k+1} <= x_max )
            
            % add matching condition
            opti.subject_to( X{k+1} == Xk_end )
                        
        % error thrown if other specification is included            
        otherwise
            error('Discretization type not supported!')       
    end
end
% add terminal cost and constraints if needed
J = J + Mterm(X{end});
if ntc > 0
    if isempty(soft_penalty)
        opti.subject_to( gf(X{end}) <= zeros(ntc,1) )
    else
        Ef = opti.variable(ntc);
        opti.subject_to( Ef >= zeros(ntc,1) )
        opti.set_initial(Ef, zeros(ntc,1))
        opti.subject_to( gf(X{end}) <= Ef )
        J = J + soft_penalty*(Ef'*Ef);
    end
end
opti.minimize( J )

% define the solver (ipopt is default)
opti.solver('ipopt',plugin_opts,solver_opts);


%% Run closed-loop simulation

% fix random seed for reproducible results 
%rng(100,'twister')

% create integrator object for simulation
dae = struct('x',x,'p',vertcat(u,w),'ode',xdotw,'quad',stageCost); % CVODES from the SUNDIALS suite
opts = struct('tf',Ts);
fsim = integrator('fsim', 'cvodes', dae, opts);

% create waitbar
f_waitbar = waitbar(0,'Running closed-loop simulations');

% loop over number of repeats
Data = cell(Nrepeat,1);
for i = 1:Nrepeat    
    % initialize table to store closed-loop simulation data
    Time = (0:Nsim)'*Ts;
    States = zeros(nx,Nsim+1)';
    Inputs = [zeros(nu,Nsim)' ; nan*ones(1,nu)];
    Disturbances = [zeros(nw,Nsim)' ; nan*ones(1,nw)];
    Objective = [zeros(1,Nsim)' ; nan];
    TimeCPU = [zeros(1,Nsim)' ; nan];
    Data{i} = table(Time, States, Inputs, Disturbances, Objective, TimeCPU);
    
    % loop over the simulation time
    Xcl = x0;
    Jcl = 0;
    Data{i}.States(1,:) = Xcl;
    for k = 1:Nsim
        % solve optimization given the most recent state
        tic
        opti.set_value(X{1}, Xcl)
        try
            sol = opti.solve();
            Ucl = sol.value(U{1});

            % warm start by updating initial condition of optimization with
            % previous solution
            opti.set_initial(sol.value_variables());
        catch
            Ucl = opti.debug.value(U{1});
        end
        Data{i}.TimeCPU(k,:) = toc;
                
        % randomly draw most recent disturbances
        Wcl = wrange.*(2*rand(nw,1)-1);
        
        % simulate system and update state
        res = fsim('x0', Xcl, 'p', vertcat(Ucl,Wcl));
        Xcl = full(res.xf);
        Jcl = Jcl + (-full(res.qf));
        
        % store data in table
        Data{i}.States(k+1,:) = Xcl;
        Data{i}.Inputs(k,:) = Ucl;
        Data{i}.Disturbances(k,:) = Wcl;
        Data{i}.Objective(k,:) = Jcl;        
    end    
    
    % update waitbar
    waitbar(i/Nrepeat,f_waitbar)
end

% close waitbar
close(f_waitbar);

% Save closed-loop data
if ~isempty(save_cl)
    save(save_cl, 'Data')
end

% Extract objective and constraints
fvals = [];
cvals = [];
for i = 1:Nrepeat
    fvals = [fvals ; Data{i}.Objective(end-1)];
    cvals = [cvals ; max([Data{i}.States(2:end,5) - 326 ; 322 - Data{i}.States(2:end,5)])];
end
cvals_indicator = (cvals < 0); % convert to 1 everywhere constraint satisfied
f = -mean(fvals);
c = [mean(cvals) ; -(mean(cvals_indicator) - 1) - 0.1];


%% Post-processing and plotting

if plot_on == 1    
    % plot the states
    for k = 5:5
        figure; hold on;
        for i = 1:Nrepeat
            plot(Data{i}.Time, Data{i}.States(:,k), '-b', 'linewidth', 2)
        end
        set(gcf,'color','w');
        set(gca,'FontSize',20)
        ylabel(['x' num2str(k)])
        if k == 5
            plot([Data{1}.Time(1), Data{1}.Time(end)], [322, 322], '--r', 'linewidth', 2)
            plot([Data{1}.Time(1), Data{1}.Time(end)], [326, 326], '--r', 'linewidth', 2)
        end
    end
    xlabel('time')
    
    % plot the inputs
    figure; hold on;
    for k = 1:nu
        subplot(nu,1,k); hold on;
        for i = 1:Nrepeat
            stairs(Data{i}.Time, Data{i}.Inputs(:,k), '-b', 'linewidth', 2)
        end
        set(gcf,'color','w');
        set(gca,'FontSize',20)
        ylabel(['u' num2str(k)])
        plot([Data{1}.Time(1), Data{1}.Time(end)], [u_min(k), u_min(k)], '--r', 'linewidth', 2)
        plot([Data{1}.Time(1), Data{1}.Time(end)], [u_max(k), u_max(k)], '--r', 'linewidth', 2)
    end
    xlabel('time')
    
    % plot the objective
    figure; hold on;
    for i = 1:Nrepeat
        stairs(Data{i}.Time, Data{i}.Objective, '-b', 'linewidth', 2)
    end
    set(gcf,'color','w');
    set(gca,'FontSize',20)
    ylabel('objective')
    xlabel('time')
end

end
