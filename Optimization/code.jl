# =================================================================================== #
# ======================       Non-Linear Decision Rules       ====================== #
# =================================================================================== #
# Open REPL in VS Code -> Ctrl+Shift+p

using Distributions, Random
using Plots
using LinearAlgebra
using JuMP
using HiGHS, Gurobi
const GRB_ENV = Gurobi.Env()

# ======================     Parâmetros do Problema   ======================== #

Dist_y = Normal(3,2.8);
Dist_X = Uniform(-2,5);
nX = 5;
nSamples = 50;

y = rand(Dist_y,nSamples);
X = rand(Dist_X,nSamples,nX);


println("\n\n\n\n");



# =================================================================================== #

# =================================================================================== #
# ==============================     MSE - JuMP     ================================ #
# =================================================================================== #

println("Start -- Mean Squared Error -- JuMP");

MSE_JuMP = Model(() -> Gurobi.Optimizer(GRB_ENV));
set_optimizer_attribute(MSE_JuMP, "OutputFlag", 0);
set_optimizer_attribute(MSE_JuMP, "NonConvex", 2);

# ========== Variáveis de Decisão ========== #

@variable(MSE_JuMP, β[1:nX]);
@variable(MSE_JuMP, β0);
@variable(MSE_JuMP, ε[1:nSamples]);

# ========== Restrições ========== #

#@constraint(MSE_JuMP, [ω ∈ 1:nSamples], ε[ω] == (y[ω] - sum(β[i]*X[ω,i] for i ∈ 1:nX))*(y[ω] - sum(β[i]*X[ω,i] for i ∈ 1:nX)));
@constraint(MSE_JuMP, [ω ∈ 1:nSamples], ε[ω] == (y[ω] - β0 - sum(β[i]*X[ω,i] for i ∈ 1:nX)));

# ========== Função Objetivo ========== #

@objective(MSE_JuMP, Min, sum(ε[ω]^2 for ω ∈ 1:nSamples));

optimize!(MSE_JuMP);

status_MSE      = termination_status(MSE_JuMP);

ObjFun_MSE      = JuMP.objective_value(MSE_JuMP);
β0Opt_MSE       = JuMP.value(β0);
βOpt_MSE        = Vector(JuMP.value.(β));



# =================================================================================== #

# =================================================================================== #
# ===========================     Gradient Descent     ============================== #
# =================================================================================== #

println("\n\n");
println("Start -- Gradient Descent\n");

nIter = 1000;

global SSE_Vector = zeros(nIter);
global iter = 1;
global β0_hat = rand(Normal(0,1));
global Grad_beta0 = 0.0;
global β_hat = rand(MvNormal(zeros(nX), Matrix{Float64}(I, nX, nX)));
global Grad_beta = zeros(nX);
global δ = 0.001; #OptChoice δ = 0.001

for iter ∈ 1:nIter

    println("     Iteration: ", iter);

    global Grad_beta0 = sum(2*(y[ω] - β0_hat - sum(β_hat[i]*X[ω,i] for i ∈ 1:nX))*(-1) for ω ∈ 1:nSamples);
    for j ∈ 1:nX
        global Grad_beta[j]  = sum(2*(y[ω] - β0_hat - sum(β_hat[i]*X[ω,i] for i ∈ 1:nX))*(-X[ω,j]) for ω ∈ 1:nSamples);
    end;

    global β0_hat = β0_hat - δ*Grad_beta0;
    for j ∈ 1:nX
        global β_hat[j] = β_hat[j] - δ*Grad_beta[j];
    end;

    global SSE_Vector[iter] = sum((y[ω] - β0_hat - sum(β_hat[i]*X[ω,i] for i ∈ 1:nX))*(y[ω] - β0_hat - sum(β_hat[i]*X[ω,i] for i ∈ 1:nX)) for ω ∈ 1:nSamples);

end;

βOpt_DradDesc = [β0_hat ; β_hat];



# ============================================================================================= #
# =========================              Print Results             ============================ #
# ============================================================================================= #

println("===============================================\n");
println("Mean Squared Error - Julia");
println("     -> Number Samples: ", nSamples);
println("     -> Status: ", status_MSE);
println("     -> Objective Function: ", ObjFun_MSE);
println("     -> Solution - β0: ", β0Opt_MSE);
println("     -> Solution - β:  ", βOpt_MSE);
println("\n===============================================");


println("===============================================\n");
println("Gradient Descent - Julia");
println("     -> Number Samples: ", nSamples);
println("     -> Objective Function: ", SSE_Vector[nIter]);
println("     -> Solution - β0: ", β0_hat);
println("     -> Solution: ", β_hat);
println("\n===============================================");