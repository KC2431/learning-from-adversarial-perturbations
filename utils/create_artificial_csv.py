import torch

# L0-uniform
in_dim = 10000
in_dims = (100, 500, 1000, 2500, 5000, 7500, 10000)
perturbation_constraints_along_with_in_dims = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
hidden_dim = 1000
n_sample = 1000
n_samples = (1000, 2000, 3000, 5000, 7000, 10000)
n_noise_sample = 10000
n_noise_samples = (1, 10, 100, 500, 1000, 1500, 2500, 5000, 7500, 10000)
norm = 'L0'
mode = 'uniform'
perturbation_constraint = 0.05
perturbation_constraints = (0.0001, 0.0004, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05)
seed = 5

with open(f'../SCRATCH/artificial_graph_csv/GDPR_CFE_{mode}_inp_dims_vs_acc.csv', 'w') as f:
    f.write('inp_dims,adv_acc_for_natural,noise_acc_for_natural\n')
    for inp_dim in in_dims:
        data = torch.load(f'../SCRATCH/artificial/{inp_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{inp_dim},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
in_dim=10000
with open(f'../SCRATCH/artificial_graph_csv/GDPR_CFE_{mode}_n_samples_vs_acc.csv', 'w') as f:
    f.write('natural_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_sample in n_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
n_sample=1000
with open(f'../SCRATCH/artificial_graph_csv/GDPR_CFE_{mode}_n_noise_samples_vs_acc.csv', 'w') as f:
    f.write('noise_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_noise_sample in n_noise_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_noise_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
n_noise_sample=10000
with open(f'../SCRATCH/artificial_graph_csv/GDPR_CFE_{mode}_modified_pixel_ratio_vs_acc.csv', 'w') as f:
    f.write('perturbation_constraints, adv_acc_for_natural, noise_acc_for_natural\n')
    for perturbation_constraint in perturbation_constraints:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{perturbation_constraint},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
"""
# L2 - Uniform
in_dim = 10000
in_dims = (100, 500, 1000, 2500, 5000, 7500, 10000)
perturbation_constraints_along_with_in_dims = (0.078, 0.17, 0.24, 0.39, 0.55, 0.67, 0.78)
hidden_dim = 1000
n_sample = 1000
n_samples = (1000, 2000, 3000, 5000, 7000, 10000)
n_noise_sample = 10000
n_noise_samples = (1, 10, 100, 500, 1000, 1500, 2500, 5000, 7500, 10000)
norm = 'L2'
mode = 'uniform'
perturbation_constraint = 0.78
perturbation_constraints = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.78)
seed = 5

norm='L2'
perturbation_constraint=0.78

with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_inp_dims_vs_acc.csv', 'w') as f:
    f.write('inp_dims,adv_acc_for_natural,noise_acc_for_natural\n')
    for inp_dim, perturbation_constraint_along in zip(in_dims,perturbation_constraints_along_with_in_dims):
        data = torch.load(f'../SCRATCH/artificial/{inp_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint_along}_{seed}')
        f.write(f"{inp_dim},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

in_dim=10000
perturbation_constraint=0.78
                
with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_n_samples_vs_acc.csv', 'w') as f:
    f.write('natural_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_sample in n_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

n_sample=1000

with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_n_noise_samples_vs_acc.csv', 'w') as f:
    f.write('noise_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_noise_sample in n_noise_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_noise_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

n_noise_sample=10000

with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_perturbation_constraints_vs_acc.csv', 'w') as f:
    f.write('perturbation_constraints, adv_acc_for_natural, noise_acc_for_natural\n')
    for perturbation_constraint in perturbation_constraints:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{perturbation_constraint},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

# Linf-Uniform
in_dim = 10000
in_dims = (100, 500, 1000, 2500, 5000, 7500, 10000)
perturbation_constraints_along_with_in_dims = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
hidden_dim = 1000
n_sample = 1000
n_samples = (1000, 2000, 3000, 5000, 7000, 10000)
n_noise_sample = 10000
n_noise_samples = (1, 10, 100, 500, 1000, 1500, 2500, 5000, 7500, 10000)
norm = 'Linf'
mode = 'uniform'
perturbation_constraint = 0.03
perturbation_constraints = (0.001, 0.005, 0.01, 0.015, 0.02, 0.03)
seed = 5

norm='Linf'
perturbation_constraint=0.03

with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_inp_dims_vs_acc.csv', 'w') as f:
    f.write('inp_dims,adv_acc_for_natural,noise_acc_for_natural\n')
    for inp_dim in in_dims:
        data = torch.load(f'../SCRATCH/artificial/{inp_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{inp_dim},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

in_dim=10000
                
with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_n_samples_vs_acc.csv', 'w') as f:
    f.write('natural_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_sample in n_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

n_sample=1000

with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_n_noise_samples_vs_acc.csv', 'w') as f:
    f.write('noise_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_noise_sample in n_noise_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_noise_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

n_noise_sample=10000

with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_perturbation_constraints_vs_acc.csv', 'w') as f:
    f.write('perturbation_constraints, adv_acc_for_natural, noise_acc_for_natural\n')
    for perturbation_constraint in perturbation_constraints:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{perturbation_constraint},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

perturbation_constraint=0.03
"""
# L0 - Gaussian
in_dim = 10000
in_dims = (100, 500, 1000, 2500, 5000, 7500, 10000)
perturbation_constraints_along_with_in_dims = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
hidden_dim = 1000
n_sample = 1000
n_samples = (1000, 2000, 3000, 5000, 7000, 10000)
n_noise_sample = 10000
n_noise_samples = (1, 10, 100, 500, 1000, 1500, 2500, 5000, 7500, 10000)
norm = 'L0'
mode = 'gauss'
perturbation_constraint = 0.05
perturbation_constraints = (0.0001, 0.0004, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05)
seed = 2

norm='L0'
perturbation_constraint=0.05

with open(f'../SCRATCH/artificial_graph_csv/GDPR_CFE_{mode}_inp_dims_vs_acc.csv', 'w') as f:
    f.write('inp_dims,adv_acc_for_natural,noise_acc_for_natural\n')
    for inp_dim in in_dims:
        data = torch.load(f'../SCRATCH/artificial/{inp_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{inp_dim},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
in_dim=10000
with open(f'../SCRATCH/artificial_graph_csv/GDPR_CFE_{mode}_n_samples_vs_acc.csv', 'w') as f:
    f.write('natural_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_sample in n_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
n_sample=1000
with open(f'../SCRATCH/artificial_graph_csv/GDPR_CFE_{mode}_n_noise_samples_vs_acc.csv', 'w') as f:
    f.write('noise_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_noise_sample in n_noise_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_noise_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
n_noise_sample=10000      
with open(f'../SCRATCH/artificial_graph_csv/GDPR_CFEE_{mode}_modified_pixel_ratio_vs_acc.csv', 'w') as f:
    f.write('perturbation_constraints, adv_acc_for_natural, noise_acc_for_natural\n')
    for perturbation_constraint in perturbation_constraints:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{perturbation_constraint},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()


"""
# L2-Gaussian
in_dim = 10000
in_dims = (100, 500, 1000, 2500, 5000, 7500, 10000)
perturbation_constraints_along_with_in_dims = (0.078, 0.17, 0.24, 0.39, 0.55, 0.67, 0.78)
hidden_dim = 1000
n_sample = 1000
n_samples = (1000, 2000, 3000, 5000, 7000, 10000)
n_noise_sample = 10000
n_noise_samples = (1, 10, 100, 500, 1000, 1500, 2500, 5000, 7500, 10000)
norm = 'L2'
mode = 'gauss'
perturbation_constraint = 0.78
perturbation_constraints = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.78)
seed = 2

norm='L2'
perturbation_constraint=0.78

with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_inp_dims_vs_acc.csv', 'w') as f:
    f.write('inp_dims,adv_acc_for_natural,noise_acc_for_natural\n')
    for inp_dim, perturbation_constraint_along in zip(in_dims,perturbation_constraints_along_with_in_dims):
        data = torch.load(f'../SCRATCH/artificial/{inp_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint_along}_{seed}')
        f.write(f"{inp_dim},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

in_dim=10000
perturbation_constraint=0.78
                
with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_n_samples_vs_acc.csv', 'w') as f:
    f.write('natural_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_sample in n_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

n_sample=1000

with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_n_noise_samples_vs_acc.csv', 'w') as f:
    f.write('noise_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_noise_sample in n_noise_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_noise_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()

n_noise_sample=10000

with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_perturbation_constraints_vs_acc.csv', 'w') as f:
    f.write('perturbation_constraints, adv_acc_for_natural, noise_acc_for_natural\n')
    for perturbation_constraint in perturbation_constraints:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{perturbation_constraint},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()


# Linf-Gaussian
in_dim = 10000
in_dims = (100, 500, 1000, 2500, 5000, 7500, 10000)
perturbation_constraints_along_with_in_dims = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
hidden_dim = 1000
n_sample = 1000
n_samples = (1000, 2000, 3000, 5000, 7000, 10000)
n_noise_sample = 10000
n_noise_samples = (1, 10, 100, 500, 1000, 1500, 2500, 5000, 7500, 10000)
norm = 'Linf'
mode = 'gauss'
perturbation_constraint = 0.03
perturbation_constraints = (0.001, 0.005, 0.01, 0.015, 0.02, 0.03)
seed = 2

norm='Linf'
perturbation_constraint=0.03

with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_inp_dims_vs_acc.csv', 'w') as f:
    f.write('inp_dims,adv_acc_for_natural,noise_acc_for_natural\n')
    for inp_dim in in_dims:
        data = torch.load(f'../SCRATCH/artificial/{inp_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{inp_dim},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
in_dim=10000
with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_n_samples_vs_acc.csv', 'w') as f:
    f.write('natural_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_sample in n_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
n_sample=1000
with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_n_noise_samples_vs_acc.csv', 'w') as f:
    f.write('noise_samples, adv_acc_for_natural, noise_acc_for_natural\n')
    for n_noise_sample in n_noise_samples:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{n_noise_sample},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
n_noise_sample=10000
with open(f'../SCRATCH/artificial_graph_csv/{norm}_{mode}_perturbation_constraints_vs_acc.csv', 'w') as f:
    f.write('perturbation_constraints, adv_acc_for_natural, noise_acc_for_natural\n')
    for perturbation_constraint in perturbation_constraints:
        data = torch.load(f'../SCRATCH/artificial/{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}')
        f.write(f"{perturbation_constraint},{data['adv_acc_for_natural']},{data['noise_acc_for_natural']}\n")
    f.close()
"""

