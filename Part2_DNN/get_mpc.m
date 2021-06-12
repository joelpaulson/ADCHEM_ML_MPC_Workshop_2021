function mpc_data = get_mpc(mpc_data)

% import casadi
import casadi.*

% parse inputs
N = mpc_data.N;
nx = mpc_data.nx;
nu = mpc_data.nu;
x = mpc_data.x;
u = mpc_data.u;
Ts = mpc_data.Ts;
xdot = mpc_data.xdot;
stageCost = mpc_data.stageCost;
terminalCost = mpc_data.terminalCost;
discretization = mpc_data.discretization;
sparse = mpc_data.sparse;
d = mpc_data.d;
orthogonal_polynomials = mpc_data.orthogonal_polynomials;
x_min = mpc_data.x_min;
x_max = mpc_data.x_max;
u_min = mpc_data.u_min;
u_max = mpc_data.u_max;
pathConstraints = mpc_data.pathConstraints;
terminalConstraints = mpc_data.terminalConstraints;
soft_penalty = mpc_data.soft_penalty;
x_init = mpc_data.x_init;
u_init = mpc_data.u_init;
plugin_opts = mpc_data.plugin_opts;
solver_opts = mpc_data.solver_opts;

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
Var = [];
Var0 = [];
for k = 1:N
    % new variable for the control input
    U{k} = opti.variable(nu);
    opti.subject_to( u_min <= U{k} <= u_max )
    opti.set_initial(U{k}, u_init)
    Var = [Var ; U{k}];
    Var0 = [Var0 ; u_init];

    % add path constraints
    if npc > 0
        if isempty(soft_penalty)
            opti.subject_to( g(X{k},U{k}) <= zeros(npc,1) )            
        else
            E{k} = opti.variable(npc);
            opti.subject_to( E{k} >= zeros(npc,1) )
            opti.set_initial(E{k}, zeros(npc,1))
            Var = [Var ; E{k}];
            Var0 = [Var0 ; zeros(npc,1)];
            opti.subject_to( g(X{k},U{k}) <= E{k} )
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
                    Var = [Var ; X{k+1}];
                    Var0 = [Var0 ; x_init];
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
            Var = [Var ; X{k+1}];
            Var0 = [Var0 ; x_init];
            
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
                Var = [Var ; Xc{k,j}];
                Var0 = [Var0 ; x_init];
                opti.subject_to( x_min <= Xc{k,j} <= x_max )
                if npc > 0
                    if isempty(soft_penalty)
                        opti.subject_to( g(Xc{k,j},U{k}) <= zeros(npc,1) )
                    else
                        Ec{k,j} = opti.variable(npc);
                        opti.subject_to( Ec{k,j} >= zeros(npc,1) )
                        opti.set_initial(Ec{k,j}, zeros(npc,1))
                        Var = [Var ; Ec{k,j}];
                        Var0 = [Var0 ; zeros(npc,1)];
                        opti.subject_to( g(Xc{k,j},U{k}) <= Ec{k,j} )
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
            Var = [Var ; X{k+1}];
            Var0 = [Var0 ; x_init];
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
        Var = [Var ; Ef];
        Var0 = [Var0 ; zeros(ntc,1)];
        opti.subject_to( gf(X{end}) <= Ef )
        J = J + soft_penalty*(Ef'*Ef);
    end
end
opti.minimize( J )

% define the solver (ipopt is default)
opti.solver('ipopt',plugin_opts,solver_opts);

% store arguments
mpc_data.opti = opti;
mpc_data.X = X;
mpc_data.U = U;
mpc_data.Var = Var;
mpc_data.Var0 = Var0;

end