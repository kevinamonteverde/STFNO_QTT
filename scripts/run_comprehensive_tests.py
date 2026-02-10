#!/usr/bin/env python3
"""
Comprehensive timing tests for TT vs QTT contractions.
Tests multiple configurations on both CPU and GPU.
"""

import sys
sys.path.insert(0, '/global/u2/p/pepi/STFNO_QTT')

import torch
import json
from tt_contraction_qtt import compare_tt_qtt_full_timing

def run_all_tests():
    """Run comprehensive tests across multiple configurations and devices."""

    configurations = [
        {"shape": (16, 16, 16, 16, 16), "batch": 2, "rank": 4},
        {"shape": (32, 32, 32, 32, 32), "batch": 4, "rank": 4},
        {"shape": (64, 64, 64, 64, 64), "batch": 2, "rank": 4},
    ]

    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    all_results = {}

    for device_name in devices:
        print("\n" + "=" * 80)
        print(f"TESTING ON {device_name.upper()}")
        print("=" * 80)

        device_results = {}

        for config in configurations:
            shape = config["shape"]
            batch = config["batch"]
            rank = config["rank"]

            shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}"

            print(f"\n{'='*80}")
            print(f"Configuration: shape={shape}, batch={batch}, rank={rank}")
            print(f"{'='*80}\n")

            # Set device
            if device_name == "cuda":
                torch.cuda.set_device(0)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Temporarily override the device in tt_contraction_qtt module
            import tt_contraction_qtt
            original_device = tt_contraction_qtt.device
            tt_contraction_qtt.device = device

            try:
                results = compare_tt_qtt_full_timing(
                    weight_shape=shape,
                    batch_size=batch,
                    rank=rank,
                    n_warmup=2,
                    n_runs=5,
                    dtype=torch.cfloat,
                )

                # Store results
                device_results[shape_key] = {
                    'config': config,
                    'tt_mean_ms': sum(results['tt_times']) / len(results['tt_times']) * 1000,
                    'tt_min_ms': min(results['tt_times']) * 1000,
                    'qtt_spatial_mean_ms': sum(results['qtt_spatial_times']) / len(results['qtt_spatial_times']) * 1000,
                    'qtt_spatial_min_ms': min(results['qtt_spatial_times']) * 1000,
                    'qtt_full_mean_ms': sum(results['qtt_full_times']) / len(results['qtt_full_times']) * 1000,
                    'qtt_full_min_ms': min(results['qtt_full_times']) * 1000,
                    'tt_params': results['tt_params'],
                    'qtt_spatial_params': results['qtt_spatial_params'],
                    'qtt_full_params': results['qtt_full_params'],
                    'dense_params': results['dense_params'],
                    'tt_detailed_avg': results['tt_detailed_avg'],
                    'qtt_spatial_detailed_avg': results['qtt_spatial_detailed_avg'],
                    'qtt_full_detailed_avg': results['qtt_full_detailed_avg'],
                }

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                device_results[shape_key] = {'error': str(e)}

            finally:
                # Restore original device
                tt_contraction_qtt.device = original_device

        all_results[device_name] = device_results

    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 80)

    for device_name in devices:
        print(f"\n{'='*80}")
        print(f"{device_name.upper()} RESULTS")
        print(f"{'='*80}\n")

        device_results = all_results[device_name]

        for shape_key, result in device_results.items():
            if 'error' in result:
                print(f"{shape_key}: ERROR - {result['error']}")
                continue

            print(f"\nConfiguration: {shape_key}")
            print(f"  Dense parameters: {result['dense_params']:,}")
            print(f"  TT parameters: {result['tt_params']:,} ({result['dense_params']/result['tt_params']:.2f}x compression)")
            print(f"  QTT Spatial parameters: {result['qtt_spatial_params']:,} ({result['dense_params']/result['qtt_spatial_params']:.2f}x compression)")
            print(f"  QTT Full parameters: {result['qtt_full_params']:,} ({result['dense_params']/result['qtt_full_params']:.2f}x compression)")
            print(f"\n  Timing (mean/min in ms):")
            print(f"    TT:          {result['tt_mean_ms']:8.3f} / {result['tt_min_ms']:8.3f}")
            print(f"    QTT Spatial: {result['qtt_spatial_mean_ms']:8.3f} / {result['qtt_spatial_min_ms']:8.3f}")
            print(f"    QTT Full:    {result['qtt_full_mean_ms']:8.3f} / {result['qtt_full_min_ms']:8.3f}")
            print(f"\n  Speedup vs TT (based on min time):")
            print(f"    QTT Spatial: {result['qtt_spatial_min_ms']/result['tt_min_ms']:.2f}x")
            print(f"    QTT Full:    {result['qtt_full_min_ms']/result['tt_min_ms']:.2f}x")

            # Detailed timing breakdown
            print(f"\n  Detailed timing breakdown (% of total):")
            print(f"    TT:")
            for key, val in result['tt_detailed_avg'].items():
                pct = val / result['tt_mean_ms'] * 100
                print(f"      {key:20s}: {val:8.4f} ms ({pct:5.2f}%)")

            print(f"    QTT Spatial:")
            for key, val in result['qtt_spatial_detailed_avg'].items():
                pct = val / result['qtt_spatial_mean_ms'] * 100
                print(f"      {key:20s}: {val:8.4f} ms ({pct:5.2f}%)")

            print(f"    QTT Full:")
            for key, val in result['qtt_full_detailed_avg'].items():
                pct = val / result['qtt_full_mean_ms'] * 100
                print(f"      {key:20s}: {val:8.4f} ms ({pct:5.2f}%)")

    # Save results to JSON
    output_file = '/global/homes/p/pepi/STFNO_QTT/scripts/comprehensive_test_results.json'

    # Convert results to JSON-serializable format
    json_results = {}
    for device_name, device_results in all_results.items():
        json_results[device_name] = {}
        for shape_key, result in device_results.items():
            json_results[device_name][shape_key] = {
                k: (v if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()})
                for k, v in result.items()
            }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_all_tests()
