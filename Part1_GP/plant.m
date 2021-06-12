function Xnext = plant(X,U,W)

A = [0.9, 0.7 ; 0, 0.8];
B = [1 ; 1];
Bd = [1 ; 0];

gXU = 2.5*(1-cos(1.6*X(1)./pi));

Xnext = A*X + B*U + Bd*(gXU + W);

end