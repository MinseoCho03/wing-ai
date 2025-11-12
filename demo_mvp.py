# demo_mvp.py
import argparse
import json
import numpy as np
from typing import Dict
from wing_ai.pipeline import WINGAIPipeline
from data.demo_data import DEMO_DATA

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj

def save_graph_json(result: Dict, filename: str = "graph.json"):
    edges_list = []
    for (source, target), data in result['edges'].items():
        edge_data = {
            'source': source,
            'target': target,
            **convert_to_serializable(data)
        }
        edges_list.append(edge_data)

    nodes_list = []
    for keyword, importance in result['nodes'].items():
        nodes_list.append({'id': keyword, 'importance': float(importance)})

    graph_data = {
        'nodes': nodes_list,
        'edges': edges_list,
        'metadata': {'total_nodes': len(nodes_list), 'total_edges': len(edges_list)}
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Graph saved to {filename}")

def choose_mode_cli() -> tuple[str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "investment"], help="Select pipeline mode")
    parser.add_argument("--main", help="Main keyword (required in investment mode)", default=None)
    parser.add_argument("--out", help="Output json filename (optional)")
    args = parser.parse_args()

    mode = args.mode
    if not mode:
        ans = input("Select mode [normal/investment]: ").strip().lower()
        mode = "investment" if ans.startswith("i") else "normal"

    main_kw = args.main
    if mode == "investment" and not main_kw:
        main_kw = input("Enter MAIN keyword (e.g., 테슬라): ").strip()

    out = args.out if args.out else f"graph_{mode}.json"
    return mode, main_kw, out

if __name__ == "__main__":
    print("=" * 60)
    print("WING AI Pipeline - MVP Demo")
    print("=" * 60)

    mode, main_kw, out_file = choose_mode_cli()
    pipeline = WINGAIPipeline(config_path="config.yaml")

    print(f"\n[Mode: {mode}]")
    if mode == "investment":
        print(f"[Main Keyword] {main_kw}")

    result = pipeline.process(
        DEMO_DATA,
        mode=mode,
        main_keyword=main_kw
    )
    save_graph_json(result, out_file)

    print("\n" + "=" * 60)
    print(f"📊 Results Summary ({mode.capitalize()} Mode)")
    print("=" * 60)

    if mode == "normal":
        print(f"\n📍 Nodes ({len(result['nodes'])}개):")
        for keyword, importance in result['nodes'].items():
            print(f"  - {keyword}: {importance:.3f}")

        print(f"\n🔗 Edges ({len(result['edges'])}개):")
        for (source, target), data in list(result['edges'].items())[:5]:
            print(f"  - {source} ↔ {target}: weight={data['weight']:.3f}")

    else:  # investment
        print(f"\n🔗 Edges with Sentiment ({len(result['edges'])}개):")
        for (source, target), data in list(result['edges'].items())[:8]:
            sentiment = data.get('sentiment_label', 'N/A')
            score = data.get('sentiment_score', 0)
            subj = data.get('sentiment_subject', None)
            deriv = data.get('sentiment_derivation', None)
            hops = data.get('hops_to_main', None)
            extra = f", hops={hops}" if hops is not None else ""
            print(f"  - {source} ↔ {target}: {sentiment} ({score:.3f}) [subject={subj}, {deriv}{extra}]")

    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)
