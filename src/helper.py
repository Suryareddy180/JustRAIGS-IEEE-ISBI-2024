=import json
import os
import tempfile
from pathlib import Path
from pprint import pprint

import SimpleITK as sitk
from PIL import Image

DEFAULT_GLAUCOMATOUS_FEATURES = {
    "appearance neuroretinal rim superiorly": None,
    "appearance neuroretinal rim inferiorly": None,
    "retinal nerve fiber layer defect superiorly": None,
    "retinal nerve fiber layer defect inferiorly": None,
    "baring of the circumlinear vessel superiorly": None,
    "baring of the circumlinear vessel inferiorly": None,
    "nasalization of the vessel trunk": None,
    "disc hemorrhages": None,
    "laminar dots": None,
    "large cup": None,
}

def inference_tasks(input_folder="test/input", output_folder="output"):
    input_files = [x for x in Path(input_folder).rglob("*") if x.is_file()]
    print("Input Files:")
    pprint(input_files)

    os.makedirs(output_folder, exist_ok=True)

    justification_stack = []

    def save_prediction(is_referable_glaucoma, likelihood_referable_glaucoma, glaucomatous_features=None):
        features = {**DEFAULT_GLAUCOMATOUS_FEATURES, **(glaucomatous_features or {})}
        justification_stack.append(features)

    for file_path in input_files:
        if file_path.suffix == ".mha":
            yield from single_file_inference(file_path, save_prediction)
        elif file_path.suffix == ".tiff":
            yield from stack_inference(file_path, save_prediction)

    write_glaucomatous_features(justification_stack, output_folder)

def single_file_inference(image_file, callback):
    with tempfile.TemporaryDirectory() as temp_dir:
        image = sitk.ReadImage(image_file)
        output_path = Path(temp_dir) / "image.jpg"
        sitk.WriteImage(image, str(output_path))
        yield output_path, callback

def stack_inference(stack, callback):
    de_stacked_images = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with Image.open(stack) as tiff_image:
            for page_num in range(tiff_image.n_frames):
                tiff_image.seek(page_num)
                output_path = Path(temp_dir) / f"image_{page_num + 1}.jpg"
                tiff_image.save(output_path, "JPEG")
                de_stacked_images.append(output_path)
                print(f"De-Stacked {output_path}")
        for image in de_stacked_images:
            yield image, callback

def write_glaucomatous_features(result, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    output_path = Path(output_folder) / "multiple-justification-binary.json"
    with open(output_path, "w") as f:
        f.write(json.dumps(result))
    print(f"Glaucomatous features written to {output_path}")