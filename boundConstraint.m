%This function is used for L-SHADE bound checking 
function vi = boundConstraint (vi, pop, lu)
% lu是一个size = 2*D的矩阵，第一行全为-100(下边界)第二行全为100(上边界)
% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound
%

[NP, D] = size(pop);  % the population size and the problem's dimension

%% check the lower bound
xl = repmat(lu(1, :), NP, 1);  % 复制矩阵，xl为size = NP*D，value = -100

pos = vi < xl;
vi(pos) = (pop(pos) + xl(pos)) / 2;

%% check the upper bound
xu = repmat(lu(2, :), NP, 1);  % xu为size = NP*D，value = 100
pos = vi > xu;
vi(pos) = (pop(pos) + xu(pos)) / 2;