import os
import itertools
from multiprocessing import Pool
import time
import os

abs_path = "/mnt/buda-perf-dataset/"

model_configs = {
    "bert": {"task":["na", "text_classification", "keyword_extraction"], "config": ["tiny", "base", "large"]},
    "deit": {"task": ["image_classification"], "config": ["base", "small"]},
    "falcon": {"task": ["na"], "config": ["7b", "7b-instruct"]},
    "flant5": {"task":["text_classification", "text_summarization"], "config": ["small", "base", "large"]},
    "flant5_past_cache_enc_dec": {"task":["na", "text_classification", "text_summarization"], "config": ["small", "base", "large"]},
    "hrnet": {"task": ["image_classification"], "config": ["w18", "v2_w18", "v2_w30", "v2_w32", "v2_w40", "v2_w44", "v2_w48", "v2_w64"]},
    "inception_v4": {"task": ["image_classification"], "config": ["224"]},
    "mobilenetv1": {"task": ["image_classification"], "config": ["192", "224"]},
    "mobilenetv2": {"task": ["image_classification"], "config": ["224", "160", "96"]},
    "mobilenetv2_timm": {"task": ["na"], "config": ["1"]},
    "mobilenetv3":  {"task": ["image_classification"], "config": ["sm", "lg"]},
    "open_pose": {"task": ["post_estimation"], "config": ["2d", "3d"]},
    "resnet": {"task": ["image_classification"], "config": ["resnet18", "resnet50", "resnet101"]},
    "stable_diffusion": {"task": ["image_generation"], "config": ["v1-4"]},
    "t5": {"task":["na", "text_classification", "keyword_extraction"], "config": ["small", "base", "large"]},
    "t5_past_cache_enc_dec": {"task":["na", "text_classification", "keyword_extraction"], "config": ["small", "base", "large"]},
    "unet": {"task":["segmentation"], "config": ["256"]},
    "vit": {"task": ["image_classification"], "config": ["base", "large"]},
    "vovnet_v1": {"task": ["image_classification"], "config": ["27s", "39", "57"]},
    "vovnet_v2": {"task": ["image_classification"], "config": ["19", "39"]},# "99"]},
    "whisper": {"task":["na", "asr"], "config": ["tiny", "base", "small", "medium", "large"]},
    "whisper_enc_dec":  {"task":["na", "asr"], "config": ["tiny", "base", "small", "medium", "large"]},
    "yolo_v5": {"task":["object_detection"], "config": ["s"]}
}

choices_dict = {
    "dataformat": ["Fp16", "Fp16_b", "Bfp8_b", "Bfp4_b"],
    # "acc_dataformat": ["Fp32", "Fp16", "Fp16_b", "Bfp8", "Bfp8_b", "Bfp4", "Bfp4_b"],
    "microbatch": [1, 4, 16, 64, 256, 512],
    "math_fidelity": ["LoFi", "HiFi2", "HiFi4"],
    "backend_opt_level": [0, 2, 4],
    "trace": ["none", "verbose"],
}

# generate comibinations for model-choices
def gen_model_combs(model_configs, choices_dict):
    combinations = []
    for model, config in model_configs.items():
        for task in sorted(config["task"]):
            for cfg in sorted(config["config"]):
                combinations.append({
                    "model": model,
                    "task": task,
                    "config": cfg,
                })
    print(len(combinations))
    print(combinations)
    return combinations

def gen_choice_combs(choices_dict):
    combinations = list(itertools.product(*choices_dict.values()))
    combinations_dicts = [
        dict(zip(choices_dict.keys(), combo)) for combo in combinations
    ]
    return combinations_dicts


def create_choice_dirs(cb, prefix):
    for choice in cb:
        choice_dir = os.path.join(prefix, str(choice["microbatch"]))
        os.makedirs(choice_dir, exist_ok=True)
        dataformat_dir = os.path.join(choice_dir, choice["dataformat"])
        os.makedirs(dataformat_dir, exist_ok=True)
        math_fidelity_dir = os.path.join(dataformat_dir, choice["math_fidelity"])
        os.makedirs(math_fidelity_dir, exist_ok=True)
        backend_opt_level_dir = os.path.join(math_fidelity_dir, str(choice["backend_opt_level"]))
        os.makedirs(backend_opt_level_dir, exist_ok=True)
        trace_dir = os.path.join(backend_opt_level_dir, choice["trace"])
        os.makedirs(trace_dir, exist_ok=True)


def create_model_dirs(mb, cb, prefix="./datasets"):
    for model in mb:
        model_dir = os.path.join(prefix, model['model'])
        os.makedirs(model_dir, exist_ok=True)
        task_dir = os.path.join(model_dir, model['task'])
        os.makedirs(task_dir, exist_ok=True)
        config_dir = os.path.join(task_dir, model['config'])
        os.makedirs(config_dir, exist_ok=True)
        create_choice_dirs(cb=cb, prefix=config_dir)

def worker(mo, co):
    print(mo, co)
    try:
        # enter to the corresponding directory
        my_dir = os.path.join("./datasets", mo["model"], mo["task"], mo["config"], str(co["microbatch"]), co["dataformat"], co["math_fidelity"], str(co["backend_opt_level"]), co["trace"])

        if mo['model'] in ['stable_diffusion', 'falcon', 'whisper', 'whisper_enc_dec', 'flant5', 'flant5_past_cache_enc_dec']:
            mo['loop_count'] = 1
        else:
            mo['loop_count'] = 128
        # dump the model and choice to the json
        with open(os.path.join(my_dir, "model.json"), "w") as f:
            f.write(str(mo))
        with open(os.path.join(my_dir, "choice.json"), "w") as f:
            f.write(str(co))

        # run the command
        os.system(f"cd {my_dir} && python {abs_path}/compile.py -m {mo['model']} \
                  --task {mo['task']} -c {mo['config']} --microbatch {co['microbatch']} \
                  --dataformat {co['dataformat']} --math_fidelity {co['math_fidelity']} \
                  --backend_opt_level {co['backend_opt_level']} --trace {co['trace']} \
                  --loop_count {mo['loop_count']} --save_output --save_tti ./tt0.tti")
        with open(f"{my_dir}/compile.sh", 'w') as f:
            f.write(f"python {abs_path}/compile.py -m {mo['model']} \
                    --task {mo['task']} -c {mo['config']} --microbatch {co['microbatch']} \
                    --dataformat {co['dataformat']} --math_fidelity {co['math_fidelity']} \
                    --backend_opt_level {co['backend_opt_level']} --trace {co['trace']} \
                    --loop_count {mo['loop_count']} --save_output --save_tti ./tt0.tti")

        with open(f"{my_dir}/run.sh", 'w') as f:
            f.write(f"python {abs_path}/benchmark.py -m {mo['model']} \
                    --task {mo['task']} -c {mo['config']} --microbatch {co['microbatch']} \
                    --dataformat {co['dataformat']} --math_fidelity {co['math_fidelity']} \
                    --backend_opt_level {co['backend_opt_level']} --trace {co['trace']} \
                    --loop_count {mo['loop_count']} --save_output --load_tti ./tt0.tti")
        
        print(f"Finished processing model {mo} with choice {co}")


    except Exception as e:
        print(f"Error processing model {mo} with choice {co}: {e}")
        return


if __name__ == "__main__":
    mb = gen_model_combs(model_configs=model_configs, choices_dict=choices_dict)
    cb = gen_choice_combs(choices_dict=choices_dict)

    create_model_dirs(mb, cb, prefix="./datasets")

    # print(len(mb), len(cb))

    pool = Pool(12)
    for jt in cb:
        for it in mb:
            pool.apply_async(worker, args=(it, jt,))

    pool.close()
    pool.join()

