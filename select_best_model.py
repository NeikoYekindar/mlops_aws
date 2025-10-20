import json
import os
import argparse
import shutil
import numpy as np

def load_test_results(results_dir):
    test_files = [
        "results_test_1.json",
        "results_test_2.json",
        "results_test_3.json",
    ]
    results = {}
    for test_file in test_files:
        path = os.path.join(results_dir, test_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing test results file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            results[test_file] = json.load(f)
    return results

def aggregate_results(test_results):
    model1_mses = []
    model1_latencies = []
    model1_path = None
    model2_mses =[]
    model2_latencies = []
    model2_path = None
    model3_mses =[]
    model3_latencies = []
    model3_path = None

    for test_name, result in test_results.items():
        if test_name == "results_test_1.json":
            model1_mses.append(result["model1"]["mse"])
            model1_latencies.append(result["model1"]["latency_sec"])
            model2_mses.append(result["model2"]["mse"])
            model2_latencies.append(result["model2"]["latency_sec"])
            model1_path = result["model1"]["path"]
            model2_path = result["model2"]["path"]
        elif test_name == "results_test_2.json":
            model1_mses.append(result["model1"]["mse"])
            model1_latencies.append(result["model1"]["latency_sec"])
            model2_mses.append(result["model2"]["mse"])
            model2_latencies.append(result["model2"]["latency_sec"])
            model1_path = result["model1"]["path"]
            model2_path = result["model2"]["path"]
        elif test_name == "results_test_3.json":
            model1_mses.append(result["model1"]["mse"])
            model1_latencies.append(result["model1"]["latency_sec"])
            model2_mses.append(result["model2"]["mse"])
            model2_latencies.append(result["model2"]["latency_sec"])
            model1_path = result["model1"]["path"]
            model2_path = result["model2"]["path"]
    
    model1_avg_mse = np.mean(model1_mses)
    model1_avg_latency = np.mean(model1_latencies)

    model2_avg_mse = np.mean(model2_mses)
    model2_avg_latency = np.mean(model2_latencies)

    return {
        "model1": {
            "path": model1_path,
            "mses": model1_mses,
            "avg_mse": model1_avg_mse,
            "latencies": model1_latencies,
            "avg_latency": model1_avg_latency
            
        },
        "model2": {
            "path": model2_path,
            "mses": model2_mses,
            "avg_mse": model2_avg_mse,
            "latencies": model2_latencies,
            "avg_latency": model2_avg_latency
        }
    }

def select_best(aggregated_results):

    model1 = aggregated_results["model1"]
    model2 = aggregated_results["model2"]

    mse1 = model1["avg_mse"]
    mse2 = model2["avg_mse"]

    lat1 = model1["avg_latency"]
    lat2 = model2["avg_latency"]
    
    if (mse1 < mse2) or (np.isclose(mse1, mse2, rtol=1e-4) and lat1 <= lat2):
        winner = "model1"
    else:
        winner = "model2"
    
    return winner, aggregated_results[winner]["path"]

def main(args):
    print("=" * 60)
    print("SELECTING BEST MODEL ACROSS ALL TESTS")
    print("=" * 60)
    print(f"\n Loading test results from: {args.results_dir}")
    test_results = load_test_results(args.results_dir)

    print("\n Aggregating results across tests...")
    aggregated = aggregate_results(test_results)

    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    for model_name in ["model1", "model2"]:
        model = aggregated[model_name]
        print(f"\n{model_name.upper()}: {model['path']}")
        print(f"  MSE per test: {[f'{mse:.6f}' for mse in model['mses']]}")
        print(f"  Average MSE:  {model['avg_mse']:.6f}")
        print(f"  Latency per test: {[f'{lat:.4f}s' for lat in model['latencies']]}")
        print(f"  Average Latency:  {model['avg_latency']:.4f}s")

    winner,best_model = select_best(aggregated)
    print(f"Winner: {winner.upper()}")
    print(f"Path: {best_model['path']}")
    print(f"Average MSE: {best_model['avg_mse']:.6f}")
    print(f"Average Latency: {best_model['avg_latency']:.4f}s")
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    final_results = {
        "test_results_summary": {
            "model1": {
                "path": aggregated["model1"]["path"],
                "mse_test_1": aggregated["model1"]["mses"][0],
                "mse_test_2": aggregated["model1"]["mses"][1],
                "mse_test_3": aggregated["model1"]["mses"][2],
                "avg_mse": aggregated["model1"]["avg_mse"],
                "avg_latency": aggregated["model1"]["avg_latency"]
            },
            "model2": {
                "path": aggregated["model2"]["path"],
                "mse_test_1": aggregated["model2"]["mses"][0],
                "mse_test_2": aggregated["model2"]["mses"][1],
                "mse_test_3": aggregated["model2"]["mses"][2],
                "avg_mse": aggregated["model2"]["avg_mse"],
                "avg_latency": aggregated["model2"]["avg_latency"]
            }
        },
        "best_model": {
            "winner": winner,
            "path": best_model["path"],
            "avg_mse": best_model["avg_mse"],
            "avg_latency": best_model["avg_latency"]
        },
        "selection_criteria": "lowest_avg_mse_across_3_tests_then_latency"
    }


    final_results_path = os.path.join(out_dir, "final_model_selection.json")
    with open(final_results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    try:
        best_model_dest = os.path.join(out_dir, "best_model.pth")
        shutil.copy2(best_model["path"], best_model_dest)
        print(f"Best model copied to: {best_model_dest}")
    except Exception as e:
        print(f"Warning: Could not copy best model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select Best Model Across Tests")
    parser.add_argument('--results-dir', type=str, required=True, help='Directory containing test results JSON files.')
    parser.add_argument('--out-dir', type=str, default=None, help='Directory to write final results and best model.')
    args = parser.parse_args()
    main(args)