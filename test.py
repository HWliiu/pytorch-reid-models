import accelerate
import torch
from prettytable import PrettyTable
from termcolor import colored
from thop import clever_format, profile

from reid_models.data import build_test_dataloaders
from reid_models.evaluate import Estimator
from reid_models.modeling import build_reid_model
from reid_models.modeling.utils import HiddenPrints
from reid_models.utils import setup_logger


def main():
    # get logger
    setup_logger(name="reid_models", log_dir="logs/test")
    logger = setup_logger(name="__main__", log_dir="logs/test")

    # get test data
    test_dataloaders = build_test_dataloaders(
        dataset_names=["dukemtmcreid", "market1501", "msmt17"]
    )

    accelerator = accelerate.Accelerator(mixed_precision="fp16")

    model_names = [
        "densenet121_abd",
        "resnet50_abd",
        "resnet50_agw",
        "resnet50_ap",
        "mlfn_dpr",
        "osnet_x1_0_dpr",
        "osnet_ain_x1_0_dpr",
        "osnet_ibn_x1_0_dpr",
        "resnet50_bot",
        "resnet50_ibn_a_bot",
        "se_resnet50_bot",
        "se_resnext50_bot",
        "senet154_bot",
        "deit_transreid",
        "vit_base_transreid",
        "vit_transreid",
        "agw_R50_fastreid",
        "agw_R50_ibn_fastreid",
        "agw_R101_ibn_fastreid",
        "agw_S50_fastreid",
        "bagtricks_mobilenet_v3_large_fastreid",
        "bagtricks_osnet_ibn_x1_0_fastreid",
        "bagtricks_osnet_x1_0_fastreid",
        "bagtricks_R50_fastreid",
        "bagtricks_SeR50_fastreid",
        "bagtricks_R50_ibn_fastreid",
        "bagtricks_R101_ibn_fastreid",
        "bagtricks_S50_fastreid",
        "bagtricks_convnext_tiny_fastreid",
        "bagtricks_densenet121_fastreid",
        "bagtricks_inception_resnet_v2_fastreid",
        "bagtricks_inception_v3_fastreid",
        "bagtricks_inception_v4_fastreid",
        "sbs_R50_fastreid",
        "sbs_R50_ibn_fastreid",
        "sbs_R101_ibn_fastreid",
        "sbs_S50_fastreid",
        "mgn_R50_fastreid"
        "mgn_R50_ibn_fastreid",
        "mgn_sbs_R50_fastreid",
        "mgn_sbs_R50_ibn_fastreid",
        "mgn_agw_R50_fastreid",
        "mgn_agw_R50_ibn_fastreid",
        "mgn_S50_fastreid",
        "mgn_S50_ibn_fastreid",
        "mgn_sbs_S50_fastreid",
        "mgn_sbs_S50_ibn_fastreid",
        "mgn_agw_S50_fastreid",
        "mgn_agw_S50_ibn_fastreid",
    ]
    for model_name in model_names:
        for dataset_name, (q_data_loader, g_data_loader) in test_dataloaders.items():
            # get model
            model = build_reid_model(model_name, dataset_name)
            q_data_loader, g_data_loader, model = accelerator.prepare(
                q_data_loader, g_data_loader, model
            )

            # evaluate metrics
            logger.info(colored(f"Evaluate {model_name} on {dataset_name}", "green"))
            estimator = Estimator(model, g_data_loader)
            cmc, mAP, mINP = estimator(q_data_loader)

            # compute macs and params
            try:
                device = next(model.parameters()).device
                input = torch.randn(1, 3, 256, 128, device=device)
                with HiddenPrints():
                    macs, params = profile(model, inputs=(input,))
                macs, params = clever_format([macs, params], "%.3f")
            except Exception as e:
                macs, params = "-", "-"

            # print results
            results = PrettyTable(
                field_names=[
                    "Dataset",
                    "Model",
                    "Top1",
                    "Top5",
                    "mAP",
                    "mINP",
                    "MACs",
                    "Params",
                ]
            )
            results.add_row(
                [
                    dataset_name,
                    model_name,
                    f"{cmc[0]:.3f}",
                    f"{cmc[4]:.3f}",
                    f"{mAP:.3f}",
                    f"{mINP:.3f}",
                    macs,
                    params,
                ]
            )
            logger.info("\n" + str(results))
            # torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
