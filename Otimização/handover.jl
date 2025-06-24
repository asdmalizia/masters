# =================================================================================== #
# =========================          Decision Rules          ======================== #
# =================================================================================== #
# Open REPL in VS Code -> Ctrl+Shift+p

using Distributions, Random
using Plots
using JuMP
using HiGHS #, Gurobi

# const GRB_ENV = Gurobi.Env();

# ======================     Parâmetros do Problema   ======================== #

u  = 150;
q  = 25;
r_ =  5;
c  = 10;

# ============================================================================ #

# ========================     Sampling Process     ========================== #

Random.seed!(1);

dmin =  50;
dmax = 150;

nCenarios = 200_000;                                # Number of Scenarios
Ω = 1:nCenarios;                                    # Set of Scenarios
p = ones(nCenarios)*(1/nCenarios);                  # Equal Probability
ξ_true = rand(Uniform(dmin, dmax), nCenarios);

println("\n\n\n\n");



# =================================================================================== #

# =================================================================================== #
# =============================     True Problem     ================================ #
# =================================================================================== #

TrueProblem = Model(HiGHS.Optimizer);
# TrueProblem = Model(() -> Gurobi.Optimizer(GRB_ENV));
# set_optimizer_attribute(TrueProblem, "OutputFlag", 0);

# ========== Variáveis de Decisão ========== #

@variable(TrueProblem, x >= 0);
@variable(TrueProblem, yS[Ω] >= 0);
@variable(TrueProblem, yR[Ω] >= 0); 

# ========== Restrições ========== #

@constraint(TrueProblem, x <= u);
@constraint(TrueProblem, [ω in Ω], yS[ω] <= ξ_true[ω]);
@constraint(TrueProblem, [ω in Ω], yS[ω] + yR[ω] <= x);

# ========== Função Objetivo ========== #

@objective(TrueProblem, Max, sum(p[ω]*(q*yS[ω] + r_*yR[ω] - c*x) for ω in Ω));

optimize!(TrueProblem);

status_True      = termination_status(TrueProblem);

ObjFun_True      = JuMP.objective_value(TrueProblem);
xOpt_True        = JuMP.value(x);
ySOpt_True       = JuMP.value.(yS);
yROpt_True       = JuMP.value.(yR);

# ============================================================================================= #
# =========================              Print Results             ============================ #
# ============================================================================================= #

println("\n\n\n\n");


println("=========================================================\n");
println("True Problem -- Number Samples: ", nCenarios, "\n");
println("     -> Status: ", status_True);
println("     -> Objective Function: ", ObjFun_True);
println("     -> First-Stage Cost: ", c*xOpt_True);
println("     -> First-Stage Solution: ", xOpt_True);
println("\n=========================================================");