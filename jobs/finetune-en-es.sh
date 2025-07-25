#!/bin/bash
#SBATCH --job-name=gpt2_finetune_eng_esp
#SBATCH --account=PAS2836
#SBATCH --output=/fs/ess/PAS2836/ipa_gpt/jobs/logs/%x-%j.out
#SBATCH --error=/fs/ess/PAS2836/ipa_gpt/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=05:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node=1

echo "===== [$(date)] JOB STARTED ====="

export BASH_ENV=/dev/null

module load cuda/12.4.1

export HF_HOME="/users/PAS2836/krishnakb/hf_cache_new"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PYTHON_EXE
unset CONDA_EXE
hash -r

export PATH="/users/PAS2836/krishnakb/ondemand/krishna_proj/cleanenv/bin:$PATH"
export VIRTUAL_ENV="/users/PAS2836/krishnakb/ondemand/krishna_proj/cleanenv"
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH"

git config --global --add safe.directory /fs/scratch/PAS2836/ipa_gpt/github/IPA_Finetuning

echo "Python: $(which python) ($(python --version))"

train_lang="both"
eval_lang="both"
for arg in "$@"; do
  case $arg in
    --train-lang=*) train_lang="${arg#*=}";;
    --eval-lang=*) eval_lang="${arg#*=}";;
    *) echo "unknown argument: $arg"; exit 1;;
  esac
done

scratch_prefix="/fs/scratch/PAS2836/ipa_gpt"
scratch_github_prefix="$scratch_prefix/github"
scratch_hf_cache_prefix="$scratch_prefix/cache"
mkdir -pv $scratch_github_prefix $scratch_hf_cache_prefix

repo_name="IPA_Finetuning"
repo_address="git@github.com:aaron-jencks/$repo_name.git"
repo_branch="trainer"
repo_dir="$scratch_github_prefix/$repo_name"
if [ ! -d "$repo_dir" ]; then
  cd "$scratch_github_prefix"
  git clone "$repo_address"
  cd "$repo_name"
  git checkout "$repo_branch"
else
  cd "$repo_dir"
  # git pull
fi

echo "===== [$(date)] RUNNING PYTHON SCRIPT ====="

mkdir -p "$scratch_prefix/checkpoints"
chmod -R u+w "$scratch_prefix/checkpoints"

TQDM_DISABLE=1 python /users/PAS2836/krishnakb/finetuning-exp.py \
  "$SLURM_JOB_ID" "xnli" \
  english_spanish_ipa_12_5_medium_50k_3epoch english_spanish_normal_12_5_medium_50k_3epoch \
  bpe-eng-spa-ipa-number-preservation bpe-eng-spa-normal-number-preservation \
  en es \
  iggy12345/xnli-en-ipa iggy12345/xnli-es-ipa \
  --lang-1-features premise hypothesis \
  --lang-2-features premise hypothesis \
  --eval-feature label label \
  --train-lang "$train_lang" \
  --eval-lang "$eval_lang" \
  --num-classes 3 \
  --is-medium \
  --training-checkpoint-prefix /users/PAS2836/krishnakb/ipa_finetune_ckpts \
  --hf-cache /users/PAS2836/krishnakb/hf_cache_new


echo "===== [$(date)] JOB COMPLETED ====="
