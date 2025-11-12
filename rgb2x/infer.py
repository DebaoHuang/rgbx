import os
import torch
import torchvision
from diffusers import DDIMScheduler
from load_image import load_exr_image, load_ldr_image
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from PIL import Image
import numpy as np


def process_images(input_folder, output_folder, seed=42, inference_step=50, num_samples=1, max_side=1000):
    os.makedirs(output_folder, exist_ok=True)

    # 加载 pipeline
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
        cache_dir=os.path.join(os.getcwd(), "model_cache"),
    ).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    required_aovs = ["albedo"]
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    # 遍历文件夹
    for filename in sorted(os.listdir(input_folder)):
        filepath = os.path.join(input_folder, filename)
        if not os.path.isfile(filepath):
            continue

        # if filename.lower().endswith(".exr"):
        #     photo = load_exr_image(filepath, tonemaping=True, clamp=True).to("cuda")
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            photo = load_ldr_image(filepath, from_srgb=True).to("cuda")
        else:
            print(f"跳过不支持的文件: {filename}")
            continue

        old_height, old_width = photo.shape[1], photo.shape[2]
        new_height, new_width = old_height, old_width

        # 调整大小，保证不超过 max_side
        ratio = old_height / old_width
        if old_height > old_width:
            new_height = max_side
            new_width = int(new_height / ratio)
        else:
            new_width = max_side
            new_height = int(new_width * ratio)

        # 保证是8的倍数
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8

        photo_resized = torchvision.transforms.Resize((new_height, new_width))(photo)

        # 生成多个样本
        for i in range(num_samples):
            for aov_name in required_aovs:
                prompt = prompts[aov_name]

                results = pipe(
                    prompt=prompt,
                    photo=photo_resized,
                    num_inference_steps=inference_step,
                    height=new_height,
                    width=new_width,
                    generator=generator,
                    required_aovs=[aov_name],
                ).images

                result = results[0][0]
                result_ori = results[1].squeeze(0).transpose(2,0,1)

                # resize 回原图大小
                result_resized = torchvision.transforms.Resize((old_height, old_width))(result)
                result_ori_resized = torchvision.transforms.Resize((old_height, old_width))(torch.from_numpy(result_ori))

                # 保存结果
                out_path = os.path.join(
                    output_folder, f"{os.path.splitext(filename)[0]}_{aov_name}.png"
                )
                result_pil = result_resized
                result_pil.save(out_path)
                print(f"保存: {out_path}")

                out_path = os.path.join(
                    output_folder, f"{os.path.splitext(filename)[0]}_{aov_name}.npy"
                )
                np.save(out_path, result_ori_resized.numpy())


if __name__ == "__main__":
    input_folder = "../IntrinsicImageDiffusion/data/synthetic/aerial_ortho_250_8000_1"   # 你要处理的文件夹
    output_folder = "./pretrained/aerial_ortho_250_8000_1"       # 输出结果保存位置

    process_images(input_folder, output_folder, seed=1234, inference_step=4, num_samples=1)
