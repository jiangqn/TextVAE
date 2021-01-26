import os

dataset = "yelp"
gpu = 3
os.system("export CUDA_VISIBLE_DEVICES=%d" % gpu)

print("train text_cnn")
os.system("python main.py --model text_cnn --task train --gpu %d --config %s_config.yaml" % (gpu, dataset))
print("test text_cnn")
os.system("python main.py --model text_cnn --task test --gpu %d --config %s_config.yaml" % (gpu, dataset))
print("train lm")
os.system("python main.py --model lm --task train --gpu %d --config %s_config.yaml" % (gpu, dataset))
print("test lm")
os.system("python main.py --model lm --task test --gpu %d --config %s_config.yaml" % (gpu, dataset))
print("train text_vae")
os.system("python main.py --model text_vae --task train --gpu %d --config %s_config.yaml" % (gpu, dataset))
print("test text_vae")
os.system("python main.py --model text_vae --task test --gpu %d --config %s_config.yaml" % (gpu, dataset))
print("compute aggregated posterior")
os.system("python main.py --model text_vae --task compute_aggregated_posterior --gpu %d --config %s_config.yaml" % (gpu, dataset))

for i in range(11):
    aggregated_posterior_ratio = i / 10
    print("aggregated_posterior_ratio: %.1f" % aggregated_posterior_ratio)
    print("register_aggregated_posterior")
    os.system("python main.py --task register_aggregated_posterior --gpu %d --config %s_config.yaml --aggregated_posterior_ratio %.1f" % (gpu, dataset, aggregated_posterior_ratio))
    print("vanilla_sample")
    os.system("python main.py --task vanilla_sample --gpu %d --config %s_config.yaml" % (gpu, dataset))
    print("get_features")
    os.system("python main.py --task get_features --gpu %d --config %s_config.yaml" % (gpu, dataset))
    print("category_analyze")
    os.system("python main.py --task category_analyze --gpu %d --config %s_config.yaml" % (gpu, dataset))
    print("length_analyze")
    os.system("python main.py --task length_analyze --gpu %d --config %s_config.yaml" % (gpu, dataset))
    print("depth_analyze")
    os.system("python main.py --task depth_analyze --gpu %d --config %s_config.yaml" % (gpu, dataset))
    print("linear_categorical_sample")
    os.system("python main.py --task linear_categorical_sample --gpu %d --config %s_config.yaml" % (gpu, dataset))
    print("linear_length_sample")
    os.system("python main.py --task linear_length_sample --gpu %d --config %s_config.yaml" % (gpu, dataset))
    print("linear_depth_sample")
    os.system("python main.py --task linear_depth_sample --gpu %d --config %s_config.yaml" % (gpu, dataset))
    print("")