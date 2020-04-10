import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import pickle

# Define global variables
natural_amino_acids = list('AGILPVFWYDERHKSTCMNQ')
hsm_predict_dir = Path("/home/zhangi/kentsislab/sh2/02_alquraishi/hsm/predict/")
domain_file = Path("input/domain_data.sh2.csv")
gene_names = ['FYN', 'SH2D1A','GRB2', 'PIK3R1_nterm', 'PIK3R1_cterm', 'PTPN11_nterm', 'PTPN11_cterm', 'BRDG1', 'SHC1']
d_scores = {} # key : peptide sequence (str), value : dict (key : target (str), value : (selectivity score (float), raw prediction scores (list of floats)))

parser = argparse.ArgumentParser(description='Generate MCMC samples')
parser.add_argument('model', type=str, help='d or id')
parser.add_argument('target', type=str, help='if single, the gene name of the target in uppercase, else "set"')
parser.add_argument('iteration', type=str, help='iteration number as a string')
args = parser.parse_args()
model = args.model
model_bool = True if args.model == 'id' else False
target = args.target
target_bool = True if args.target == 'set' else False
iteration = args.iteration

def align(peptide):
    '''
    Align peptide such that the pY has 7 leading and trailing amino acids or "-".
    Ex: AEDEyVNEPLYL -> ---AEDEyVNEPLYL
    
    Parameters
    ----------
    peptide : str
        peptide sequence
    
    Returns
    -------
    aligned_peptide : str
        aligned peptide sequence
    '''
    
    # Check that there is a pY and identify its index
    try:
        pY_index = peptide.index('y')
    except:
        raise Exception("Warning: No pY!")
    
    # Align residues before the pY
    count_before_pY =  len(peptide[:pY_index]) 
    count_after_pY =  len(peptide[pY_index+1:]) 
    aligned_peptide = ''
    if count_before_pY < 7: # Add leading "-"
        aligned_peptide += '-'*(7 - count_before_pY) + peptide[:pY_index]
    elif count_before_pY > 7: # Truncate the peptide 
        aligned_peptide = peptide[count_before_pY-7:pY_index]
    else:
        aligned_peptide = peptide[:7]
    
    # Add pY
    aligned_peptide += 'y'
    
    # Align the residues after the pY
    if count_after_pY < 7: # Add trailing "-"
        aligned_peptide += peptide[pY_index+1:] + '-'*(7 - count_after_pY)
    elif count_after_pY > 7: # Truncate the peptide 
        aligned_peptide += peptide[pY_index+1:pY_index+8]
    else:
        aligned_peptide += peptide[pY_index+1:]
        
    return aligned_peptide

def prepare_input(aligned_peptide, outfile):
    '''
    Prepare peptide input file to HSM prediction.
    
    Parameters
    ----------
    aligned_peptide : str
        aligned peptide sequence
    outfile : str
        path at which to save the input file.

    '''
    # Prep input peptide files for HSM prediction
    puniprot = ''
    ptype = 'phosphosite'
    pformat = 'FIXED'
    pd.DataFrame([[puniprot, aligned_peptide, ptype, pformat]]).to_csv(outfile, index=False, header=False)

# Run HSM prediction
def predict(domain_file, peptide_file, outfile, use_hsm_id=False):
    '''
    Run HSM prediction on peptide against representative SH2 domains.
    
    Parameters
    ----------
    domain_file : str
        path to the input peptide file
    peptide_file : str
        path to the input peptide file
    outfile : str
        path at which to save the input file
    use_hsm_id : bool
        indicates whether to use HSM/D model (default) or HSM/ID  model
    
    Returns
    -------
    int
        0 if running the prediction was a success, 1 if it failed
    ''' 
    current_dir = os.getcwd()
    os.chdir(hsm_predict_dir)
    
    if use_hsm_id:# Use HSM/ID model (instead of default HSM/D model)
        hsm_id_model_dir = Path("/home/zhangi/kentsislab/sh2/02_alquraishi/hsm/predict/models/hsm_id_pretrained/")
        command = ["python", "predict_domains.py", current_dir / domain_file , current_dir / peptide_file, "-o", current_dir / outfile, "-m", hsm_id_model_dir, "--model_format", hsm_id_model_dir / "model_formats.csv"]
    else:
        command = ["python", "predict_domains.py", current_dir / domain_file , current_dir / peptide_file, "-o", current_dir / outfile]
    try:
        subprocess.run(command, check=True)
        os.chdir(current_dir)
        os.remove(peptide_file)
        return True
    except Exception as e:
        print(f"Failed: {e.stdout}") # Note the stdout is empty
        os.chdir(current_dir)
        return False
    
# Read in prediction
def get_predictions(predictions_file, gene_names):
    '''
    Get dictionary of predictions for each SH2 domain
    
    Parameters
    ----------
    predictions_file : str
        path to predictions file (outputted by HSM)
    gene_names : list of str
        gene names corresponding to each row in the predictions file
    
    Returns
    -------
    dict
        key: SH2 domain gene name, value: prediction value
    ''' 

    df = pd.read_csv(predictions_file, names=['domain_type', 'domain_uniprot', 'domain_sequence', 'peptide_type', 'peptide_uniprot', 'peptide_sequence', 'prediction'])
    os.remove(predictions_file)
    df['gene_name'] = gene_names
    return pd.Series(df['prediction'].values,index=df['gene_name']).to_dict()

def is_valid_peptide(peptide):
    '''
    Check that the peptide contains only natural amino acids.
    
    Parameters
    ----------
    peptide : str
        peptide sequence
    
    Returns
    -------
    bool
        Indicating whether the peptide is valid
    '''
    
    for aa in peptide:
        if aa not in natural_amino_acids:
            print("Warning: Peptide contains nonnatural amino acids!")
            return False
    return True

def selectivity(peptide, target, peptide_input_dir, prediction_output_dir, use_hsm_id=False):
    '''
    Compute the selectivity score for a peptide
    
    Parameters
    ----------
    peptide : str
        peptide sequence
    target : str
        gene name of the target SH2 domain
    peptide_input_dir : str
        path to peptide input file for HSM prediction
    prediction_output_dir : str
        path to prediction file outputted by HSM
    use_hsm_id : bool
        indicates whether to use HSM/D model (default) or HSM/ID  model
    
    Returns
    -------
    float
        the selectvity score as a log unnormalized probability
    list of floats
        list of raw HSM/ID prediction scores w.r.t each target
    '''
    
    # Check to see if selectivity score was already generated
    if peptide in d_scores and target in d_scores[peptide]:
        return d_scores[peptide][target][0], d_scores[peptide][target][1]
    else: # Compute selectivity score
        # Specify file names
        peptide_file = peptide_input_dir / f"peptide_{peptide}.csv"
        predictions_file = prediction_output_dir / f"output_{peptide}.csv"

        # Prepare input for HSM prediction
        aligned_peptide = align(peptide)
        prepare_input(aligned_peptide, peptide_file)

        # Run HSM prediction
        status = predict(domain_file, peptide_file, predictions_file, use_hsm_id)
        if status:
            # Retrieve predictions and convert to Kas
            predictions = get_predictions(predictions_file, gene_names)
            raw_scores = list(predictions.values())
            kas = {sh2: prediction / (1 - prediction) for sh2, prediction in predictions.items()}

            # Save target prediction and create list of off target predictions
            target_prediction = kas[target]
            del kas[target]
            off_target_predictions = list(kas.values())
            selectivity = np.log(target_prediction) - np.log(sum(off_target_predictions))
            
            # Add peptide selectivity score to dict
            if peptide in d_scores:
                d_scores[peptide][target] = (selectivity, raw_scores)
            else:
                d_scores[peptide] = {target: (selectivity, raw_scores)}
            return selectivity, raw_scores
        else:
            raise Exception("Error: Prediction failed")

def selectivity_set(peptides, targets, peptide_input_dir, prediction_output_dir, use_hsm_id=False):
    '''
    Compute the selectivity score for a peptide set
    
    Parameters
    ----------
    peptide : str
        peptide sequence
    target : str
        gene name of the target SH2 domain
    peptide_input_dir : str
        path to peptide input file for HSM prediction
    prediction_output_dir : str
        path to prediction file outputted by HSM
    use_hsm_id : bool
        indicates whether to use HSM/D model (default) or HSM/ID  model
    
    Returns
    -------
    float
        the selectvity score as a log unnormalized probability
    list of list of floats
        for each peptide, a list of raw HSM/ID prediction scores w.r.t each target
    '''
    selectivity_scores = []
    raw_scores_set = []
    for peptide, target in zip(peptides, targets):
        selectivity_score, raw_scores = selectivity(peptide, target, peptide_input_dir, prediction_output_dir, use_hsm_id)
        selectivity_scores.append(selectivity_score)
        raw_scores_set.append(raw_scores)
    return sum(selectivity_scores), raw_scores_set

def sample_random_peptide():
    '''
    Generate a random pY peptide with 7 leading and trailing residues w.r.t. pY
    
    Returns
    -------
    str
        random peptide
    '''
    
    peptide = ''.join(np.random.choice(natural_amino_acids, size=14))
    return peptide[:7] + 'y' + peptide[7:]

def mutate_one_site_symmetric(peptide):
    positions = list(range(15))
    positions.pop(7) # Remove pY position from list of allowed positions to mutate
    position_to_mutate = np.random.choice(positions, size=1)[0]
    residue_to_insert = np.random.choice(natural_amino_acids, size=1)[0]
    peptide_as_list = list(peptide)
    peptide_as_list[position_to_mutate] = residue_to_insert
    return ''.join(peptide_as_list)

def mutate_one_site_symmetric_set(peptides):
    peptide = np.random.choice(peptides, size=1)[0]
    peptide_index = peptides.index(peptide)
    positions = list(range(15))
    positions.pop(7) # Remove pY position from list of allowed positions to mutate
    position_to_mutate = np.random.choice(positions, size=1)[0]
    residue_to_insert = np.random.choice(natural_amino_acids, size=1)[0]
    peptide_as_list = list(peptide)
    peptide_as_list[position_to_mutate] = residue_to_insert
    return peptides[:peptide_index] + [''.join(peptide_as_list)] + peptides[peptide_index + 1:]

def random_walk_mh(x0, log_prob_func, symmetric_proposal_func, n_steps=1000, is_set=False):
    """Random-walk Metropolis-Hastings
    
    Parameters
    ----------
    x0 : object
        initial state of sampler
    log_prob_func : callable, accepts an object like x0 and returns a float
        unnormalized log probability density function
    args : additional 
    symmetric_proposal_func : callable, accepts an object like x0 and returns another object like x0
        assume this samples from a symmetric proposal distribution (p(y|x) = p(x|y))
    n_steps : integer
        number of MCMC steps
    is_set : bool
        indicates whether to sample individual peptides (False) or sets of peptides (True)
    
    Returns
    -------
    traj : [n_steps + 1] list of objects
        trajectory of samples generated by MCMC
    log_probs : [n_steps + 1] array of floats
        unnormalized log-probabilities of the samples
    acceptance_fraction : float in [0,1]
        fraction of accepted proposals
    raw_scores_all : list of list of list of floats
        for each sample, a list of lists containing the raw HSM/ID prediction scores w.r.t each target and each peptide
        Note: for the list of lists, the outer list is per peptide and the inner list contains per target scores
    """
    traj = [x0]
    log_probs = []
    raw_scores_all = []
    if not is_set:
        log_prob, raw_scores = log_prob_func(x0, target, Path('/home/zhangi/kentsislab/sh2/03b_design_peptides/input'), Path('/home/zhangi/kentsislab/sh2/03b_design_peptides/output'), use_hsm_id=model_bool)
    else:
        log_prob, raw_scores = log_prob_func(x0, gene_names, Path('/home/zhangi/kentsislab/sh2/03b_design_peptides/input'), Path('/home/zhangi/kentsislab/sh2/03b_design_peptides/output'), use_hsm_id=model_bool)
    log_probs.append(log_prob)
    raw_scores_all.append(raw_scores)

    acceptances = 0
    r = range(n_steps)
    trange = tqdm(r)
    for n in trange:

        x_proposal = symmetric_proposal_func(traj[-1])
        if not is_set:
            log_prob_proposal, raw_scores = log_prob_func(x_proposal, target, Path('input'), Path('output'), use_hsm_id=model_bool)
        else:
            log_prob_proposal, raw_scores = log_prob_func(x_proposal, gene_names, Path('input'), Path('output'), use_hsm_id=model_bool)

        if np.random.rand() < np.exp(log_prob_proposal - log_probs[-1]):
            traj.append(x_proposal)
            log_probs.append(log_prob_proposal)
            raw_scores_all.append(raw_scores)
            acceptances += 1
        else:
            traj.append(traj[-1])
            log_probs.append(log_probs[-1])
            raw_scores_all.append(raw_scores_all[-1])

        trange.set_postfix({'log_prob': log_probs[-1], 'accept_fraction': float(acceptances) / (1 + n)})
    del (trange)

    return traj, np.array(log_probs), float(acceptances) / n_steps, raw_scores_all

def main():

    if not target_bool:
        x0 = sample_random_peptide()
        selectivity_fn = selectivity
        proposal_fn = mutate_one_site_symmetric
    else:
        x0 = [sample_random_peptide() for i in range(9)]
        selectivity_fn = selectivity_set
        proposal_fn = mutate_one_site_symmetric_set

    traj, log_probs, acceptance_rate, raw_scores_all = random_walk_mh(x0,
                   log_prob_func=selectivity_fn,
                   symmetric_proposal_func=proposal_fn,
                   n_steps=10000, is_set=target_bool
                  )

    write_dir = Path(f'/home/zhangi/kentsislab/sh2/03b_design_peptides/samples/mcmc_hsm_{model}_{target.lower()}')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    with open(write_dir / f'peptides_{iteration}.pkl', 'wb') as f:
        pickle.dump(traj, f)

    with open(write_dir / f'selectivity_scores_{iteration}.pkl', 'wb') as f:
        pickle.dump(log_probs, f)

    with open(write_dir / f'acceptance_{iteration}.pkl', 'wb') as f:
        pickle.dump(acceptance_rate, f)
        
    with open(write_dir / f'initial_{iteration}.pkl', 'wb') as f:
        pickle.dump(x0, f)

    with open(write_dir / f'raw_scores_{iteration}.pkl', 'wb') as f:
        pickle.dump(raw_scores_all, f)

    plt.hist(log_probs[1000:], bins=20, alpha=0.5)
    plt.xlabel('log(selectivity)')
    plt.title("Distribution of selectivity scores for MCMC-samples")
    plt.savefig(write_dir / f'plot_{iteration}.png', dpi=300)

if __name__ == "__main__":
    main()