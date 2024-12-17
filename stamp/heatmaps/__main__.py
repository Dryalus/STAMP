import argparse
from collections.abc import Collection
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
import pandas as pd
from fastai.vision.learner import Learner, load_learner
from jaxtyping import Float, Int
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from PIL import Image
from torch import Tensor

from stamp.preprocessing.helpers.common import supported_extensions

def load_slide_ext(wsi_dir: Path) -> openslide.OpenSlide:
    # Check if any supported extension matches the file
    if wsi_dir.suffix not in supported_extensions:
        raise FileNotFoundError(
            f"No supported slide file found for slide {wsi_dir.name} in provided directory {wsi_dir.parent}\
                                 \nOnly support for: {supported_extensions}"
        )
    else:
        return openslide.open_slide(wsi_dir)


def get_stride(coords: Tensor, output_dir: Path) -> int:
    print("Coords is: ", coords.shape, coords)
    coords_df= pd.DataFrame(coords)
    coords_df.to_csv(str(output_dir) + '{slide_name}_coords.csv',index=False)
    xs = coords[:, 0].unique(sorted=True)
    stride = (xs[1:] - xs[:-1]).min()
    print("Stride is: ",stride.shape,stride)
    return stride


def gradcam_per_category(
    learn: Learner, feats: Tensor, categories: Collection
) -> tuple[Tensor, Tensor]:
    feats_batch = feats.expand((len(categories), *feats.shape)).detach()
    feats_batch.requires_grad = True
    preds = torch.softmax(
        learn.model(feats_batch, torch.tensor([len(feats)] * len(categories))),
        dim=1,
    )
    preds.trace().backward()
    gradcam = (feats_batch * feats_batch.grad).mean(-1).abs()
    return preds, gradcam


def vals_to_im(
    scores: Float[Tensor, "n_tiles *d_feats"],
    norm_coords: Int[Tensor, "n_tiles *d_feats"],
    output_dir: Path
) -> Float[Tensor, "i j *d_feats"]:
    """Arranges scores in a 2d grid according to coordinates"""
    size = norm_coords.max(0).values.flip(0) + 1
    im = torch.zeros((*size, *scores.shape[1:]))

    flattened_im = im.flatten(end_dim=1)
    flattened_coords = norm_coords[:, 1] * im.shape[1] + norm_coords[:, 0]
    flattened_im[flattened_coords] = scores

    im = flattened_im.reshape_as(im)

    print("Scores is: ", scores.shape, scores)
    scores_df= pd.DataFrame(scores)
    scores_df.to_csv(str(output_dir) + 'scores.csv', index=False)

    return im


def show_thumb(slide, thumb_ax: Axes, attention: Tensor) -> None:
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    dims_um = np.array(slide.dimensions) * mpp
    thumb = slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int))
    thumb_ax.imshow(np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8])
    return np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8]


def show_class_map(
    class_ax: Axes, top_scores: Tensor, gradcam_2d, categories: Collection[str]
) -> None:
    cmap = plt.get_cmap("Pastel1")
    classes = cmap(top_scores.indices[:, :, 0])
    classes[..., -1] = (gradcam_2d.sum(-1) > 0).detach().cpu() * 1.0
    class_ax.imshow(classes)
    class_ax.legend(
        handles=[
            Patch(facecolor=cmap(i), label=cat) for i, cat in enumerate(categories)
        ]
    )


def get_n_toptiles(
    slide,
    category: str,
    output_dir: Path,
    coords: Tensor,
    scores: Tensor,
    stride: int,
    n: int = 8,
) -> None:
    slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])

    (output_dir / f"toptiles_{category}").mkdir(exist_ok=True, parents=True)

    # determine the scaling factor between heatmap and original slide
    # 256 microns edge length by default, with 224px = ~1.14 MPP (± 10x magnification)
    feature_downsample_mpp = (
        256 / stride
    )  # NOTE: stride here only makes sense if the tiles were NON-OVERLAPPING
    scaling_factor = feature_downsample_mpp / slide_mpp

    top_score = scores.topk(n)

    # OPTIONAL: if the score is not larger than 0.5, it's indecisive on directionality
    # then add [top_score.values > 0.5]
    top_coords_downscaled = coords[top_score.indices]
    top_coords_original = np.uint(top_coords_downscaled * scaling_factor)

    # NOTE: target size (stride, stride) only works for NON-OVERLAPPING tiles
    # that were extracted in previous steps.
    for score_idx, pos in enumerate(top_coords_original):
        tile = (
            slide.read_region(
                (pos[0], pos[1]),
                0,
                (np.uint(stride * scaling_factor), np.uint(stride * scaling_factor)),
            )
            .convert("RGB")
            .resize((stride, stride))
        )
        tile.save(
            (output_dir / f"toptiles_{category}")
            / f"score_{top_score.values[score_idx]:.2f}_toptiles_{category}_{(pos[0], pos[1])}.jpg"
        )


def main(
    slide_name: str,
    feature_dir: Path,
    wsi_dir: Path,
    model_path: Path,
    output_dir: Path,
    n_toptiles: int = 8,
    overview: bool = True,
) -> None:
    learn = load_learner(model_path)
    learn.model.eval()
    categories: Collection[str] = learn.dls.train.dataset._datasets[
        -1
    ].encode.categories_[0]

    # for h5_path in feature_dir.glob(f"**/{slide_name}.h5"):
    for slide_path in wsi_dir.glob(f"**/{slide_name}.*"):
        h5_path = feature_dir / slide_path.with_suffix(".h5").name
        slide_output_dir = output_dir / h5_path.stem
        slide_output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Creating heatmaps for {slide_path.name}...")
        with h5py.File(h5_path) as h5:
            feats = torch.tensor(h5["feats"][:]).float()
            coords = torch.tensor(h5["coords"][:], dtype=torch.int)

        # stride is 224 using normal operations
        stride = get_stride(coords, output_dir)

        preds, gradcam = gradcam_per_category(
            learn=learn, feats=feats, categories=categories
        )
        gradcam_2d = vals_to_im(gradcam.permute(-1, -2), torch.div(coords, stride, rounding_mode='floor'), output_dir).detach()

        scores = torch.softmax(
            learn.model(feats.unsqueeze(-2), torch.ones((len(feats)))), dim=1
        )
        # get row data for qupath visualization
        print("Gradcam from main is: ", gradcam.shape, gradcam)
        gradcam_df= pd.DataFrame(gradcam)
        gradcam_df.to_csv(slide_output_dir /f'gradcam-{slide_path.name}.csv', index=False)

        print("Scores from main is: ", scores.shape, scores)
        scores_df= pd.DataFrame(scores)
        scores_df.to_csv(slide_output_dir /f'scores-{slide_path.name}.csv', index=False)

        print("Coords from main is: ", coords.shape, coords)
        coords_df= pd.DataFrame(coords)
        coords_df.to_csv(slide_output_dir /f'coords-{slide_path.name}.csv', index=False)

        # Create csv-file for qupath visualization

        #--------SCORES---------------------------    
        
        # Werte aus den CSV-Dateien extrahieren
        column1 = coords_df.iloc[:, 0]
        column2 = coords_df.iloc[:, 1]
        column5 = scores_df.iloc[:, 0]
        #column6 = scores_df.iloc[:, 1]
        #column7 = scores_df.iloc[:, 2]
        #column8 = scores_df.iloc[:, 2]

        # Konstante Werte entsprechend pixel der Tiles
        column3 = [224] * len(column1)
        column4 = [224] * len(column1)

        array = [column1, column2, column3, column4, column5]
        
        csvcat = ['x','y','width','height']
        for i, cat in enumerate(categories):
            print(i)
            print(cat)
            csvcat.append(cat)
            array.append(coords_df.iloc[:,i])
        
        print(csvcat)
        array = np.array(array)

        df_qupath_scores = pd.DataFrame(array, columns=csvcat)
        # DataFrame mit Spalten erstellen --> Labels für Spalte 5-x
        #df_qupath_scores = pd.DataFrame({
            #'x': column1,
            #'y': column2,
            #'width': column3,
            #'height': column4,
            #'BaSq': column5,
            #'Lum': column6,
            #'Stroma': column7,
            #POLE': column8                                        
        #})

        df_qupath_scores.to_excel(slide_output_dir /f'qupath_scores-{slide_path.name}.xlsx', index=False)
        df_qupath_scores.to_csv(slide_output_dir /f'qupath_scores-{slide_path.name}.csv', index=False)

        #-------GRADCAM--------------------------
         # Werte aus den CSV-Dateien extrahieren
        g_column1 = coords_df.iloc[:, 0]
        g_column2 = coords_df.iloc[:, 1]
        g_column5 = gradcam_df.iloc[:, 0]
        g_column6 = gradcam_df.iloc[:, 1]
        g_column7 = gradcam_df.iloc[:, 2]
        #g_column8 = gradcam_df.iloc[:, 3]

        # Konstante Werte entsprechend pixel der Tiles
        g_column3 = [224] * len(g_column1)
        g_column4 = [224] * len(g_column1)

        # DataFrame mit Spalten erstellen --> Labels für Spalte 5-x
        df_qupath_gradcam = pd.DataFrame({
            'x': g_column1,
            'y': g_column2,
            'width': g_column3,
            'height': g_column4,
            'BaSq': g_column5,
            'Lum': g_column6,
            'Stroma': g_column7,
            #'POLE': g_column8
        })

        df_qupath_gradcam.to_excel(slide_output_dir /f'qupath_gradcam-{slide_path.name}.xlsx', index=False)
        df_qupath_gradcam.to_csv(slide_output_dir /f'qupath_gradcam-{slide_path.name}.csv', index=False)

        
        scores_2d = vals_to_im(scores, torch.div(coords, stride, rounding_mode='floor'), output_dir).detach()
        fig, axs = plt.subplots(nrows=2, ncols=max(2, len(categories)), figsize=(12, 8))

        show_class_map(
            class_ax=axs[0, 1],
            top_scores=scores_2d.topk(2),
            gradcam_2d=gradcam_2d,
            categories=categories,
        )

        slide = load_slide_ext(slide_path)

        for ax, (pos_idx, category) in zip(axs[1, :], enumerate(categories)):
            ax: Axes
            topk = scores_2d.topk(2)
            category_support = torch.where(
                # To get the "positiveness",
                # it checks whether the "hot" class has the highest score for each pixel
                topk.indices[..., 0] == pos_idx,
                # Then, if the hot class has the highest score,
                # it assigns a positive value based on its difference from the second highest score
                scores_2d[..., pos_idx] - topk.values[..., 1],
                # Likewise, if it has NOT a negative value based on the difference of that class' score to the highest one
                scores_2d[..., pos_idx] - topk.values[..., 0],
            )

            # So, if we have a pixel with scores (.4, .4, .2) and would want to get the heat value for the first class,
            # we would get a neutral color, because it is matched with the second class
            # But if our scores were (.4, .3, .3), it would be red,
            # because now our class is .1 above its nearest competitor

            attention = torch.where(
                topk.indices[..., 0] == pos_idx,
                gradcam_2d[..., pos_idx] / gradcam_2d.max(),
                (
                    others := gradcam_2d[
                        ..., list(set(range(len(categories))) - {pos_idx})
                    ]
                    .max(-1)
                    .values
                )
                / others.max(),
            )

            score_im = plt.get_cmap("RdBu")(
                -category_support * attention / attention.max() / 2 + 0.5
            )

            score_im[..., -1] = attention > 0

            ax.imshow(score_im)
            ax.set_title(f"{category} {preds[0,pos_idx]:1.2f}")
            target_size=np.array(score_im.shape[:2][::-1]) * 8
            # latest PIL requires shape to be a tuple (), not array []
            Image.fromarray(np.uint8(score_im * 255)).resize(
                tuple(target_size), resample=Image.NEAREST
            ).save(
                slide_output_dir
                / f"scores-{h5_path.stem}--score_{category}={preds[0][pos_idx]:0.2f}.png"
            )

            get_n_toptiles(
                slide=slide,
                category=category,
                stride=stride,
                output_dir=slide_output_dir,
                scores=scores[:, pos_idx],
                coords=coords,
                n=n_toptiles,
            )

        if overview:
            thumb = show_thumb(
                slide=slide,
                thumb_ax=axs[0, 0],
                attention=attention,
            )
            Image.fromarray(thumb).save(slide_output_dir / f"thumbnail-{h5_path.stem}.png")

            for ax in axs.ravel():
                ax.axis("off")

            fig.savefig(slide_output_dir / f"overview-{h5_path.stem}.png")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("heatmaps")
    parser.add_argument(
        "--slide-name",
        metavar="PATH",
        type=str,
        required=True,
        help="Name of the WSI to create heatmap for (no extensions)",
    )
    parser.add_argument(
        "--wsi-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory containing the SVSs",
    )
    parser.add_argument(
        "--feature-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory containing the slides' features",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        dest="model_path",
        type=Path,
        required=True,
        help="Path to the trained model's export.pkl",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory to save the heatmaps to",
    )
    parser.add_argument(
        "--n-toptiles",
        type=int,
        default=8,
        required=False,
        help="Number of toptiles to generate, 8 by default",
    )
    parser.add_argument(
        "--overview",
        type=bool,
        default=True,
        required=False,
        help="Generate final overview image",
    )
    args = parser.parse_args()
    main(**vars(args))
