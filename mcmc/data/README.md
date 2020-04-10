# Samples

## Directory naming convention
Each directory is named as follows: `{sampling_algorithm}_{prediction_model}_{target}`

Examples:
- `mcmc-mh_hsm-id_fyn` means that 
    1) The samples were generated using Metropolis-Hastings MCMC
    2) The prediction model used was HSM/ID
    3) The target SH2 domain is a single peptide: FYN

Note: When the {target} is 'set', this means that the targets were a set of SH2 domains: FYN, SH2D1A, GRB2, PIK3R1 n term, PIK3R1 c term, PTPN11 n term, PTPN11 c term, BRDG1, SHC1 

## File naming convention
Each directory may contain the following file types:
- `acceptance_{iteration}.pkl` - acceptance rate
- `initial_{iteration}.pkl` - starting peptide
- `peptides_{iteration}.pkl`- trajectory of peptides
- `plot_{iteration}.png`- histogram of log(target)s
- `plot_{iteration}_vs_rand.png`- histogram of log(target)s for mcmc samples vs randomly generated samples
- `raw_scores_{iteration).pkl`- predicted binding affinities
- `selectivity_scores_{iteration}.pkl` - log(target)s

