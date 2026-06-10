import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import DataParallel
import os
from models import FuteFWI, InversionNet, Generator, Ablation_1, Ablation_2
from utils import parse_test_args, create_testing_dataset, evaluate, evaluate_sample, plot_vmodel


if __name__ == "__main__":
    # parse args
    args = parse_test_args()

    # determine model
    if args.model == "FuteFWI":
        if not args.ablation:
            model = FuteFWI()
        else:
            """Ablation models"""
            if args.ablation == "sfe":
                model = Ablation_1()
            elif args.ablation == "tm":
                model = Ablation_2()
            else:
                raise RuntimeError("Unexpected ablation type.")
    elif args.model == "InversionNet":
        model = InversionNet()
    elif args.model == "VelocityGAN":
        model = Generator()
    else:
        raise NotImplementedError("Unsupported model type!")

    # load model
    device = torch.device(args.device)
    print("Using device:", device)
    model = model.to(device)
    name = args.model if args.name is None else args.name
    if args.model == "FuteFWI" and args.ablation is not None:
        name = f"{name}_ablation1" if args.ablation == "sfe" else f"{name}_ablation2"
    dataset_name = f"{args.dataset}_{args.version}"
    if args.gaussian_noise:
        save_path = os.path.join(args.path, f"{name}_{dataset_name}_ND.pt")
    else:
        save_path = os.path.join(args.path, f"{name}_{dataset_name}_D.pt")
    model.load_state_dict(torch.load(save_path, map_location=device))
    if args.device == "cuda":
        model = DataParallel(model)
    model.eval()
    dataset = create_testing_dataset(dataset_name, args.gaussian_noise)

    # create sampler for plot_vmodel
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    random_sampler = RandomSampler(dataset, num_samples=1, generator=generator)

    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=64)
        dataloader_sample = DataLoader(dataset, batch_size=64, sampler=random_sampler)
        if args.sample_eval:
            evaluate_sample(dataloader_sample, model, device)
        else:
            evaluate(dataloader, model, device)
        if args.draw:
            if args.gaussian_noise:
                save_name = f"{args.model}_{dataset_name}_ND"
            else:
                save_name = f"{args.model}_{dataset_name}_D"
            plot_vmodel(dataloader_sample, model, save_name, device)
