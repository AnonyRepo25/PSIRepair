import os
import time
import argparse

from dataclasses import asdict
from typing import List, Dict, Any
from . import rs_utils as rsu
from .model import make_model
from .agents import PSIRepairAgent
from .project import (
    Project,
    get_defects4j_v1_2_0_projects,
    get_defects4j_v2_0_0_projects,
    get_vul4j_projects,
)
from .utils import (
    SkipException,
    _is_single_method_bug,
    _is_debug_mode,
)


def get_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--basic_config_file", type=str, required=True)
    parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--only_init_benchmark_projects", action="store_true", default=False)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--projects", nargs="*", default=None)
    parser.add_argument("--bugs", nargs="*", default=None)
    parser.add_argument("--disable_project_lock", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)

    # For PSIRepair
    ## For patch generation agent
    parser.add_argument("--psirepair_max_num_patches_per_bug", type=int, required=True)
    ## For context retrieval agent
    parser.add_argument("--psirepair_r_local_cache_dir", type=str, required=True)
    parser.add_argument("--psirepair_r_embedding_model_config_file", type=str, required=True)
    ### For GraphBasedCodeSnippetRetrievalAgent
    parser.add_argument("--psirepair_r_num_retrieved_subgraphs", type=int, required=True)
    parser.add_argument("--psirepair_r_num_retrieved_nodes_per_subgraph", type=int, required=True)
    parser.add_argument("--psirepair_r_num_group_of_retrieved", type=int, required=True)
    parser.add_argument("--psirepair_r_num_retrieved_nodes_per_group", type=int, required=True)
    parser.add_argument("--psirepair_r_ratio_of_retrieved_fields_nodes", type=str, required=True)
    parser.add_argument("--psirepair_r_subgraph_selection_strategy", type=str, required=True)
    parser.add_argument("--psirepair_r_num_rerank_tries_per_retrieved_node", type=int, required=True)
    ## For error resolution agent
    parser.add_argument("--psirepair_max_num_error_resolution_attempts", type=int, required=True)

    args = parser.parse_args()

    if args.bugs == ["all"]:
        args.bugs = None

    if args.overwrite:
        rsu._wlog(f"Type 'overwrite' to confirm (to remove {args.output_dir}): ", end="")
        if input().strip() != "overwrite":
            raise ValueError("Not confirmed to overwrite")
        rsu._wlog(f"Removing {args.output_dir}...")
        rsu._sp_system(f"rm -rf {args.output_dir}", logging=False)

    if not os.path.isfile(args.model_config_file):
        raise ValueError(f"Model config file not found: {args.model_config_file}")

    if not os.path.isfile(args.basic_config_file):
        raise ValueError(f"Basic config file not found: {args.basic_config_file}")
    
    basic_config = rsu._load_json(args.basic_config_file)
    if not os.path.isdir(basic_config["java7_home"]):
        raise ValueError(f"Java 7 home not found: {basic_config['java7_home']}")
    if not os.path.isdir(basic_config["java8_home"]):
        raise ValueError(f"Java 8 home not found: {basic_config['java8_home']}")
    if not os.path.isdir(basic_config["java17_home"]):
        raise ValueError(f"Java 17 home not found: {basic_config['java17_home']}")
    if not os.path.isdir(basic_config["d4j_v120_home"]):
        raise ValueError(f"Defects4j V1.2.0 home not found: {basic_config['d4j_v120_home']}")
    if not os.path.isdir(basic_config["d4j_v200_home"]):
        raise ValueError(f"Defects4j V2.0.0 home not found: {basic_config['d4j_v200_home']}")
    if not os.path.isfile(basic_config["d4j_v120_bug_info_file"]):
        raise ValueError(f"Defects4j V1.2.0 bug info file not found: {basic_config['d4j_v120_bug_info_file']}")
    if not os.path.isfile(basic_config["d4j_v200_bug_info_file"]):
        raise ValueError(f"Defects4j V2.0.0 bug info file not found: {basic_config['d4j_v200_bug_info_file']}")
    if not os.path.isfile(basic_config["vul4j_bug_info_file"]):
        raise ValueError(f"Vul4j bug info file not found: {basic_config['vul4j_bug_info_file']}")

    os.makedirs(basic_config["d4j_checkout_tmp_path"], exist_ok=True)
    os.makedirs(basic_config["vul4j_checkout_tmp_path"], exist_ok=True)

    if not os.path.isfile(args.psirepair_r_embedding_model_config_file):
        raise ValueError(f"PSIRepair R embedding model config file not found: {args.psirepair_r_embedding_model_config_file}")
    args.psirepair_r_embedding_model_config = rsu._load_json(args.psirepair_r_embedding_model_config_file)
    del args.psirepair_r_embedding_model_config_file

    args.d4j_v1_2_0_cli = f"{basic_config['d4j_v120_home']}/framework/bin/defects4j"
    args.d4j_v2_0_0_cli = f"{basic_config['d4j_v200_home']}/framework/bin/defects4j"
    args.d4j_v1_2_0_bugs_stat_file = basic_config["d4j_v120_bug_info_file"]
    args.d4j_v2_0_0_bugs_stat_file = basic_config["d4j_v200_bug_info_file"]
    args.vul4j_bug_info_file = basic_config["vul4j_bug_info_file"]

    os.environ["JAVA7_HOME"] = basic_config["java7_home"]
    os.environ["JAVA8_HOME"] = basic_config["java8_home"]
    os.environ["JAVA17_HOME"] = basic_config["java17_home"]
    os.environ["D4J_CHECKOUT_TMP_PATH"] = basic_config["d4j_checkout_tmp_path"]
    os.environ["VUL4J_CHECKOUT_TMP_PATH"] = basic_config["vul4j_checkout_tmp_path"]
    os.environ["CAPREPAIR_CLEAN_PREV_CHECKOUT"] = "0"

    return vars(args)


def _launch_psirepair(project: Project, config: Dict[str, Any]):
    output_dir = os.path.abspath(config["output_dir"])
    p_output_dir = os.path.join(output_dir, project.get_itendifier())
    p_finish_mark_filename = os.path.join(p_output_dir, "__finished__")
    model = make_model(config["model_config_file"])

    def _bug_filter(bug):
        if not _is_single_method_bug(bug):
            return False
        if config["bugs"] and bug.get_itendifier() not in config["bugs"]:
            return False
        return True

    cared_bugs = project.get_bugs(_filter=_bug_filter)
    num_cared_bugs = len(cared_bugs)
    rsu._ilog(f"Repairing {project}: {num_cared_bugs} (cared: SF, `--bugs`) bugs")
    for i, bug in enumerate(cared_bugs, start=1):
        try:
            b_start_time = time.time()
            b_output_dir = os.path.join(p_output_dir, bug.get_itendifier())
            b_results_jf = os.path.join(b_output_dir, "repair_results.json")
            b_skip_mark_jf = os.path.join(b_output_dir, "__skip__.json")

            if os.path.isfile(b_results_jf):
                rsu._wlog(f"Already repaired, skipped: {bug.get_short_name()}")
                continue

            if os.path.isfile(b_skip_mark_jf):
                rsu._wlog(f"Already ensured skipped, skipped: {bug.get_short_name()}")
                continue

            bug_flock = rsu.FileLock(lockfile=f"{output_dir}/{project.get_itendifier()}.{bug.get_itendifier()}.lock", retry=0)
            if not bug_flock.try_lock():
                rsu._wlog(f"Other process is repairing {bug.get_short_name()}, skipped")
                continue

            bug.project = project.checkout(bug, "b")  # b: bug

            rsu._ilog(f"Repairing {bug.project} ({i}/{num_cared_bugs})")
            if _is_debug_mode():
                rsu._ilog(">>>> Bug details:")
                rsu._ilog(f">>>>     project: {bug.project}")
                rsu._plog(f">>>>     bug_locations", bug.bug_locations)
                rsu._ilog(f">>>>     num_buggy_lines: {bug.num_buggy_lines}")
                rsu._ilog(f">>>>     is_single_line_bug: {bug.is_single_line_bug}")
                rsu._ilog(f">>>>     is_single_method_bug: {bug.is_single_method_bug}")
                rsu._ilog(f">>>>     is_single_file_bug: {bug.is_single_file_bug}")
                rsu._ilog(f">>>>     _interal_id: {bug._interal_id}")

            os.makedirs(b_output_dir, exist_ok=True)

            rsu._ilog(f"Generating patches for: {bug.get_short_name()}")

            psirepair_agent = PSIRepairAgent(
                model=model,
                **{k.removeprefix("psirepair_"): v for k, v in config.items() if k.startswith("psirepair_")},
            )
            result = psirepair_agent.ask(psirepair_agent.Input(bug))

            if hasattr(result, "generated_patches"):
                rsu._ilog(f"Generated {len(result.generated_patches)} patches for: {bug.get_short_name()}")

            if getattr(result, "found_plausible", False):
                rsu._ilog(f"Found one plausible patch for: {bug.get_short_name()}")

            if result is not None:
                b_end_time = time.time()
                result_dict = asdict(result)
                result_dict["start_time"] = b_start_time
                result_dict["end_time"] = b_end_time
                result_dict["duration"] = b_end_time - b_start_time
                rsu._save_as_json(result_dict, b_results_jf)

        except SkipException as ex:
            rsu._wlog(f"Skip bug: {bug.get_short_name()}")
            rsu._wlog(f"Reason: {ex}")
            rsu._save_as_json({"skip_reason": str(ex)}, b_skip_mark_jf)
            continue
        finally:
            print("finally")
            try:
                bug_flock.unlock()
            except:
                pass

    if config["bugs"] is not None:
        rsu._wlog(f"Don't mark {project.get_itendifier()} as finished because of `--bugs`")
    elif config["disable_project_lock"]:
        rsu._wlog(f"Don't mark {project.get_itendifier()} as finished because of `--disable_project_lock`")
    else:
        rsu._save_as_json(None, p_finish_mark_filename)


def repair_projects(projects: List[Project], config: Dict[str, Any]):

    launch_experiment = _launch_psirepair

    if config["projects"] is not None:
        projects = [p for p in projects if p.get_itendifier() in config["projects"]]
    rsu._ilog(f"Selected projects: {[p.get_itendifier() for p in projects]}")

    for proj in projects:
        if config["only_init_benchmark_projects"]:
            rsu._ilog(f"Initializing project: {proj}")
            for bug in proj.get_bugs():
                bug_snapshot = proj.checkout(bug, "b")  # b: bug
                rsu._ilog(f"Initialized {bug_snapshot} for {bug.get_short_name()}")
        else:
            output_dir = config["output_dir"]
            disable_project_lock = config["disable_project_lock"]
            os.makedirs(output_dir, exist_ok=True)
            proj_output_dir = os.path.join(output_dir, proj.get_itendifier())
            proj_finish_mark_filename = os.path.join(proj_output_dir, "__finished__")
            proj_lock = f"{output_dir}/{proj.get_itendifier()}.lock"
            if not disable_project_lock:
                with rsu.FileLock(proj_lock, retry=0) as flock:
                    if flock.locked and not os.path.exists(proj_finish_mark_filename):
                        rsu._ilog(f"\033[34m>>=== Repairing project {proj} ===<<\033[0m")
                        launch_experiment(project=proj, config=config)
                    else:
                        rsu._wlog(f">>==== Skipped project {proj} ====<<")
                        rsu._wlog(f">>>> Saved to (or Working on) {proj_output_dir}")
            elif not os.path.exists(proj_finish_mark_filename):
                rsu._ilog(f"\033[34m>>=== Repairing project {proj} ===<<\033[0m")
                launch_experiment(project=proj, config=config)


def main(config: Dict[str, Any]) -> int:
    rsu._ilog("=============== [PSIRepair] ===============")
    rsu._plog("Config", config)

    if config["only_init_benchmark_projects"]:
        rsu._wlog("Only init benchmark projects")

    available_benchmarks = ["d4j-v120", "d4j-v200", "vul4j"]
    if config["benchmarks"] == ["all"]:
        rsu._wlog("Use all available benchmarks")
        config["benchmarks"] = available_benchmarks
    elif not all(b in available_benchmarks for b in config["benchmarks"]):
        raise ValueError(f"Unknown benchmark: {config['benchmarks']}")

    projects = []

    if "d4j-v120" in config["benchmarks"]:
        rsu._ilog("Loading Defects4J v1.2.0 projects")
        projects += get_defects4j_v1_2_0_projects(
            d4j_cli=config["d4j_v1_2_0_cli"],
            d4j_bugs_stat_file=config["d4j_v1_2_0_bugs_stat_file"],
        )

    if "d4j-v200" in config["benchmarks"]:
        rsu._ilog("Loading Defects4J v2.0.0 projects")
        projects += get_defects4j_v2_0_0_projects(
            d4j_cli=config["d4j_v2_0_0_cli"],
            d4j_bugs_stat_file=config["d4j_v2_0_0_bugs_stat_file"],
        )

    if "vul4j" in config["benchmarks"]:
        rsu._ilog("Loading Vul4J projects")
        projects += get_vul4j_projects(
            bug_info_file=config["vul4j_bug_info_file"],
        )

    rsu._plog("Projects", projects)

    # Make output dir
    os.makedirs(config["output_dir"], exist_ok=True)
    # Save config as a json file
    config_file = os.path.join(config["output_dir"], "__config.json")
    rsu._save_as_json(config, filename=config_file)

    # Start repairing projects
    repair_projects(projects, config=config)
