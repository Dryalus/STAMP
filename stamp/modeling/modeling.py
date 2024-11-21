import argparse
from pathlib import Path

from .marugoto.transformer.helpers import train_categorical_model_, deploy_categorical_model_, categorical_crossval_

def main():
    parser = argparse.ArgumentParser(
        description='Associative modeling with a Vision Transformer.')
    
    parser.add_argument("--clini_table", type=Path, help="Path to clini_excel file")
    parser.add_argument("--slide_table", type=Path, help="Path to slide_table file")
    parser.add_argument("--feature_dir", type=Path, help="Path to feature directory")
    parser.add_argument("--output_path", type=Path, help="Path to output file")
    parser.add_argument("--target_label", type=str, help="Target label")
    parser.add_argument("--cat_labels", type=str, nargs="+", default=[], help="Category labels")
    parser.add_argument("--cont_labels", type=str, nargs="+", default=[], help="Continuous labels")
    parser.add_argument("--categories", type=str, nargs="+", default=None, help="Categories")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--one_model", action="store_true", help="Run full training instead of cross-validation")
    group.add_argument("--deploy_model", type=Path, help="Path to the model .pkl to deploy")
    group.add_argument("--n_splits", type=int, default=5, help="Number of splits")
    group.add_argument("--dimension", type=int, default=512, help="Dimensions of model")
    group.add_argument("--depth", type=int, default=2, help="Number of Layers")
    group.add_argument("--heads", type=int, default=8, help="Heads")
    group.add_argument("--mlp_dimension", type=int, default=512, help="mlp Dimensions of model")
    group.add_argument("--dropout", type=float, default=.0, help="which percent gets droptout")

    
    args = parser.parse_args()

    if args.one_model:
        #run full training for 1 model
        train_categorical_model_(clini_table=args.clini_table, 
                                 slide_table=args.slide_table,
                                 feature_dir=args.feature_dir, 
                                 output_path=args.output_path,
                                 target_label=args.target_label, 
                                 cat_labels=args.cat_labels,
                                 cont_labels=args.cont_labels, 
                                 categories=args.categories)
    elif args.deploy_model:
        #deploy 1 model on data
        deploy_categorical_model_(clini_table=args.clini_table,
                                  slide_table=args.slide_table,
                                  feature_dir=args.feature_dir,
                                  model_path=args.deploy_model,
                                  output_path=args.output_path,
                                  target_label=args.target_label,
                                  cat_labels=args.cat_labels,
                                  cont_labels=args.cont_labels,
                                  transMilDim=args.dimension,
                                  transMilDepth=args.depth, 
                                  transMilheads=args.heads, 
                                  transMilMlp_dim=args.mlp_dimension, 
                                  transMilDropout=args.dropout)

    else:
        #run cross validation for n_splits models
        categorical_crossval_(clini_table=args.clini_table, 
                              slide_table=args.slide_table,
                              feature_dir=args.feature_dir,
                              output_path=args.output_path,
                              target_label=args.target_label,
                              cat_labels=args.cat_labels,
                              cont_labels=args.cont_labels,
                              categories=args.categories,
                              n_splits=args.n_splits,
                              transMilDim=args.dimension,
                              transMilDepth=args.depth, 
                              transMilheads=args.heads, 
                              transMilMlp_dim=args.mlp_dimension, 
                              transMilDropout=args.dropout)


if __name__ == "__main__":
    main()
