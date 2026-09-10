import os
import argparse
import json
import logging
from dotenv import load_dotenv
from graphs.cyber_graph import build_cyber_graph
from tools.retrieve_mitre import retrieve_mitre_db

logger = logging.getLogger("cyber_agent")


from tools.dpo_engine import generate_good_dataset, populate_bad_dataset, load_log_targets, derive_default_dataset_path


def main(args):
    # Configure tiered logging levels based on verbosity count (-v vs -vv)
    if args.verbose >= 2:
        app_log_level = logging.DEBUG
        third_party_level = logging.DEBUG
    elif args.verbose == 1:
        app_log_level = logging.DEBUG
        third_party_level = logging.WARNING
    else:
        app_log_level = logging.INFO
        third_party_level = logging.WARNING

    logging.basicConfig(
        level=app_log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Configure third-party loggers accordingly
    for lib in ("httpx", "httpcore", "chromadb", "urllib3", "google", "langchain"):
        logging.getLogger(lib).setLevel(third_party_level)

    # -------------------------------------------------------------
    # Mode 1: DPO Dataset Generation (Phase 'good' or 'bad')
    # -------------------------------------------------------------
    if args.data_gen == "good":
        if not args.logs:
            raise ValueError("Error: --logs <file_or_directory> is required when running --data-gen good")
        
        logger.info("=== Starting DPO Dataset Generation (Phase 1: 'good' chosen generation) ===")
        out_path = generate_good_dataset(logs_path=args.logs, dataset_output_path=args.dataset)
        print("\n" + "=" * 60)
        print(f"🎉 DPO Phase 1 Complete! 'Chosen' responses saved to:\n👉 {out_path}")
        print("Next step: Run Phase 2 with:")
        print(f"python app.py --data-gen bad --dataset {out_path}")
        print("=" * 60)
        return

    if args.data_gen == "bad":
        target_dataset = args.dataset
        if not target_dataset:
            if args.logs:
                target_dataset = derive_default_dataset_path(args.logs)
            else:
                raise ValueError("Error: Please provide --dataset <path_to_dataset.json> (or --logs) for --data-gen bad")
        
        logger.info("=== Starting DPO Dataset Population (Phase 2: 'bad' rejected population) ===")
        out_path = populate_bad_dataset(dataset_path=target_dataset)
        print("\n" + "=" * 60)
        print(f"🎉 DPO Phase 2 Complete! Full (prompt, chosen, rejected) dataset saved to:\n👉 {out_path}")
        print("=" * 60)
        return

    # -------------------------------------------------------------
    # Mode 2: Standard CyberAgent SOC Triage Workflow
    # -------------------------------------------------------------
    if not args.logs:
        raise ValueError("Error: --logs <path_to_log_or_folder> is required to run the triage workflow.")

    logger.info("Initializing CyberAgent environment...")
    
    # Ensure MITRE knowledge base is available
    retrieve_mitre_db(force_reload=args.reload_mitre_db)

    # Build cyber agent graph
    cyber_graph = build_cyber_graph()

    log_targets = load_log_targets(args.logs)
    logger.info("Running CyberAgent triage workflow for %d log target(s)...", len(log_targets))

    for idx, (filepath, logs) in enumerate(log_targets, 1):
        filename = os.path.basename(filepath)
        print("\n" + "=" * 55)
        print(f"🔍 [{idx}/{len(log_targets)}] Triage Analysis for: {filename}")
        print("=" * 55)

        initial_state = {
            "raw_logs": logs,
            "search_query": "",
            "retrieved_chunks": [],
            "triage_result": None,
            "retry_count": 0,
            "error_msg": ""
        }
        
        result = cyber_graph.invoke(initial_state)
        triage_result = result.get("triage_result")
        
        if triage_result:
            print("\n🛡️  TRIAGE ASSESSMENT RESULT:")
            if hasattr(triage_result, "model_dump_json"):
                print(triage_result.model_dump_json(indent=2))
            else:
                print(triage_result)
        else:
            print(f"[!] Triage failed. Error: {result.get('error_msg')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberAgent SOC Log Triage & DPO Dataset Generator")
    parser.add_argument("--logs", type=str, required=False, default=None, help="Path to a log file or directory of logs")
    parser.add_argument("--config", type=str, required=False, default="config/config.yaml", help="Path to the config file")
    parser.add_argument("--reload-mitre-db", action="store_true", help="Force reload of the MITRE ATT&CK vector database")
    parser.add_argument("--data-gen", choices=["good", "bad"], default=None, help="Run DPO dataset generator: 'good' for chosen responses, 'bad' for rejected responses")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the DPO dataset JSON file to load/save")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity level: -v for application debug, -vv for third-party HTTP/Chroma debug")
    args = parser.parse_args()
    
    load_dotenv()
    main(args)




    