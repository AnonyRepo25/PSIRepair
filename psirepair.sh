#! /bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_dir> [additional_args]"
    exit 1
fi

this_dir=$(dirname "$0")
output_dir=$(realpath "$1")
additional_args=${@:2}

basic_config_file=${this_dir}/configs/basic_config.json
model_config_file=${this_dir}/configs/llm_config.json

psirepair_max_num_patches_per_bug=45
psirepair_r_local_cache_dir=${this_dir}/cache/context_retrieval
psirepair_r_embedding_model_config_file=${this_dir}/configs/embedding_model_config.json
psirepair_r_num_retrieved_subgraphs=1
psirepair_r_num_retrieved_nodes_per_subgraph=12
psirepair_r_num_group_of_retrieved=4
psirepair_r_num_retrieved_nodes_per_group=3
psirepair_r_ratio_of_retrieved_fields_nodes=auto
psirepair_r_ratio_of_retrieved_fields_nodes_to_show=${psirepair_r_ratio_of_retrieved_fields_nodes}
psirepair_r_subgraph_selection_strategy=contextual_and_similarity
psirepair_r_num_rerank_tries_per_retrieved_node=1
psirepair_max_num_error_resolution_attempts=2 # 5 x 3 x (2 + 1) = 45

timestamp=$(date +"%Y%m%d%H%M%S")
log_file="${output_dir}/__log_d${timestamp}.txt"
echo "Log file: ${log_file}"
echo "Ouput dir: ${output_dir}"

mkdir -p ${output_dir}
export ERA_NRNpSG=9 # 3x3, r_num_retrieved_nodes_per_subgraph for ERA
export ERA_SGSS=only_max_subgraph # r_subgraph_selection_strategy for ERA
python -u -m PSIRepair \
    --basic_config_file ${basic_config_file} \
    --model_config_file ${model_config_file} \
    --psirepair_max_num_patches_per_bug ${psirepair_max_num_patches_per_bug} \
    --psirepair_r_local_cache_dir ${psirepair_r_local_cache_dir} \
    --psirepair_r_embedding_model_config_file ${psirepair_r_embedding_model_config_file} \
    --psirepair_r_num_retrieved_subgraphs ${psirepair_r_num_retrieved_subgraphs} \
    --psirepair_r_num_retrieved_nodes_per_subgraph ${psirepair_r_num_retrieved_nodes_per_subgraph} \
    --psirepair_r_num_group_of_retrieved ${psirepair_r_num_group_of_retrieved} \
    --psirepair_r_num_retrieved_nodes_per_group ${psirepair_r_num_retrieved_nodes_per_group} \
    --psirepair_r_ratio_of_retrieved_fields_nodes ${psirepair_r_ratio_of_retrieved_fields_nodes} \
    --psirepair_r_subgraph_selection_strategy ${psirepair_r_subgraph_selection_strategy} \
    --psirepair_r_num_rerank_tries_per_retrieved_node ${psirepair_r_num_rerank_tries_per_retrieved_node} \
    --psirepair_max_num_error_resolution_attempts ${psirepair_max_num_error_resolution_attempts} \
    --output_dir ${output_dir} \
    ${additional_args} \
    2>&1 | tee ${log_file}
