from utils.lib import *
from utils.dist import dist_init
HUGGING_FACE_DIR = {
    "bert-base-uncased": './_models/huggingface_transformers/bert-base-uncased/' # need to download first.
}


def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_with_config(parsed_args):
    """This function will set args based on the input config file.
    (1) it only overwrites unset parameters,
        i.e., these parameters not set from user command line input
    (2) it also sets configs in the config file but declared in the parser
    """
    # convert to EasyDict object,
    # enabling access from attributes even for nested config
    # e.g., args.train_datasets[0].name
    args = edict(vars(parsed_args))
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                         if arg.startswith("--")}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


class Args(object):
    """Shared options for pre-training and downstream tasks.
    For each downstream task, implement a get_*_args function,
    see `get_pretraining_args()`

    Usage:
    >>> shared_configs = SharedConfigs()
    >>> pretraining_config = shared_configs.get_pretraining_args()
    """

    def __init__(self, desc="shared config"):
        parser = argparse.ArgumentParser(description=desc)
        # path configs
        parser.add_argument(
            "--data_dir", default='./_datasets', type=str, required=False,
            help="Directory with all datasets, each in one subfolder")
        parser.add_argument(
            "--txt_dir", default='', type=str, required=False,
            help="Directory with all txt data")
        parser.add_argument(
            "--img_tsv_dir", default='', type=str, required=False,
            help="Directory with all img data")
        parser.add_argument(
            "--dataset", default='', type=str, required=False, nargs="+",
            help="which datasets to use")
        parser.add_argument(
            "--data_ratio", type=float,
            default=1.0)
        parser.add_argument(
            "--path_output", default='./_snapshot/', type=str, required=False,
            help="The output directory to save checkpoint and test results.")

        # model
        parser.add_argument(
            "--reinit_head", type=str_to_bool, nargs='?',
            const=True, default=False)

        # vision backbone
        parser.add_argument(
            "--vis_backbone", type=str, default='vidswin',
            choices=['vidswin'])
        parser.add_argument(
            "--vis_backbone_size", type=str, default='base',
            choices=['base'])
        parser.add_argument(
            "--kinetics", type=int, default=-1, choices=[600, 400, -1])
        parser.add_argument(
            "--vis_backbone_init", type=str, default='2d',
            choices=['3d'])

        # text backbone
        parser.add_argument(
            "--txt_backbone", type=str, default='bert-base-uncased',
            choices=['bert-base-uncased'])
        parser.add_argument(
            "--txt_backbone_embed_only", type=str_to_bool, nargs='?',
            const=False, default=True)

        # fusion encoder
        parser.add_argument(
            "--fusion_encoder", type=str, default='bert-base-uncased',
            choices=['bert-base-uncased'])
        parser.add_argument(
            "--fusion_encoder_rand_init",
            type=str_to_bool, nargs='?',
            const=True, default=False)

        # training configs
        parser.add_argument("--n_workers", default=4, type=int,
                            help="number of  workers")
        parser.add_argument("--size_batch", default=8, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--size_img", default=224, type=int,
                            help="image input size")
        parser.add_argument("--size_frame", default=4, type=int,
                            help="frame input length")
        parser.add_argument("--max_size_frame", default=6, type=int,
                            help="max size frame for temporal embedding")
        parser.add_argument("--max_size_patch", default=14, type=int,
                            help="max size patch for spatial embedding")
        parser.add_argument("--size_patch", default=32, type=int,
                            help="image_patch size")
        parser.add_argument("--size_vocab", default=-1, type=int,
                            help="number of answers for open-ended QA")
        parser.add_argument("--size_txt_pre", default=25, type=int,
                            help="text input length as pretext")
        parser.add_argument(
            "--img_transform", default=["img_rand_crop"], type=str,
            nargs='+',
            choices=["pad_resize", "img_rand_crop",
                     "vid_rand_crop", "img_center_crop"],
            required=False, help="img transform")
        parser.add_argument("--size_txt", default=25, type=int,
                            help="text input length")
        parser.add_argument("--lr", default=1.2e-5, type=float,
                            help="learning rate")
        parser.add_argument("--decay", default=1e-3, type=float,
                            help="Weight deay.")
        parser.add_argument("--size_epoch", default=20, type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument('--seed', type=int, default=88,
                            help="random seed for initialization.")
        parser.add_argument('--logging_steps', type=int, default=20,
                            help="log memory usage per X steps")
        parser.add_argument("--vis_backbone_lr_mul", default=1, type=float,
                            help="visual backbone lr multiplier")
        parser.add_argument("--max_grad_norm", default=-1, type=float,
                            help="Max gradient norm.")
        parser.add_argument('--deepspeed',
                            help="use deepspeed",  type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--use_checkpoint', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument("--temp", default=1, type=float,
                            help="temperature used in vtm/odr and retrieval")
        parser.add_argument("--local_rank", type=int, default=0,
                            help="For distributed training.")

        # pretrain
        parser.add_argument("--size_part", default=8, type=int,
                            help="number of parts for pretraining data")
        parser.add_argument(
            "--pretrain_tasks", default=["mtm", "vtm"],
            type=str, nargs="+",
            choices=["mtm", "vtm"],
            required=False, help="pretraining tasks")
        parser.add_argument("--p_mask", default=0.15, type=float,
                            help="mask prob")
        parser.add_argument(
            "--enable_task_token", type=str_to_bool,
            nargs='?', const=True, default=False)
        parser.add_argument(
            "--task_token", default=None, type=str,
            choices=["vtm", "mc", "oe", "cap"], required=False,
            help="task tokens used in finetuining")
        parser.add_argument(
            "--enable_prompt", type=str_to_bool,
            nargs='?', const=True, default=False)
        parser.add_argument(
            '--mask_pos', default='append', type=str,
            choices=["append", "prepend", "insert", "replace"], required=False,
            help="where to put [MASK]")
        # ========= resume training or load pre_trained weights
        parser.add_argument(
            '--path_ckpt', type=str, default='',
            help="pretrained ckpt")

        # retrieval test
        parser.add_argument('--multi_clip_testing', type=str_to_bool,
                            nargs='?', const=True, default=False)

        # can use config files, will only overwrite unset parameters
        parser.add_argument("--config", help="JSON config files")
        self.parser = parser

    def parse_args(self):
        parsed_args = self.parser.parse_args()
        args = parse_with_config(parsed_args)
        if os.path.exists(args.path_ckpt):
            args.vis_backbone_init = 'random'
            filename, _ = os.path.splitext(args.path_ckpt.split("/")[-1])

        if args.type != "pretrain":
            del args.size_part
            del args.pretrain_tasks
            args.txt_dir = args.data_dir
            args.img_tsv_dir = args.data_dir

        if args.type != "retrieval":
            del args.multi_clip_testing
            args.task_token = "vtm"

        if args.type != "qaoe":
            del args.size_vocab

        if args.type not in ["qamc", "qaoe"]:
            del args.reinit_head
        else:
            del args.temp
        if args.txt_backbone in HUGGING_FACE_DIR:
            pretrained_dir = HUGGING_FACE_DIR[args.txt_backbone]
            if os.path.exists(pretrained_dir):
                args.txt_backbone = HUGGING_FACE_DIR[args.txt_backbone]
            else:
                print("No pre-trained txt backbone for" +
                      f" {args.txt_backbone}, will download on-the-fly")
        else:
            print("Did not find pre-trained txt backbone for" +
                  f" {args.txt_backbone}, will download on-the-fly")
        args.tokenizer = args.txt_backbone

        if args.fusion_encoder in HUGGING_FACE_DIR:
            pretrained_dir = HUGGING_FACE_DIR[args.fusion_encoder]
            if os.path.exists(pretrained_dir):
                args.fusion_encoder = HUGGING_FACE_DIR[args.fusion_encoder]
            else:
                print("No pre-trained fusion encoder for" +
                      f" {args.fusion_encoder}, will download on-the-fly")
        else:
            print("Did not find pre-trained fusion encoder for" +
                  f" {args.fusion_encoder}, will download on-the-fly")

        return args


sharedArgs = Args()


def get_args(distributed=True):
    args = sharedArgs.parse_args()
    dist_init(args, distributed)

    if not args.distributed:
        args.deepspeed = False

    args.effective_batch_size = args.size_batch * args.num_gpus
    if os.path.exists(args.path_ckpt):
        path_ckpt_dir = os.path.dirname(args.path_ckpt)
        training_args = f"{path_ckpt_dir}/args.json"
        if os.path.exists(training_args):
            args = update_args(args)
    return args


def update_args(args):
    path_ckpt_dir = os.path.dirname(args.path_ckpt)
    training_args = edict(json.load(open(f"{path_ckpt_dir}/args.json", "r")))

    print("===============Loaded model training args=================")
    print(f"\t\t{json.dumps(training_args)}")
    print("===============Default args=================")
    print(f"\t\t{json.dumps(args)}")
    toUpdate = [
        "vis_backbone", "vis_backbone_size", 
        "kinetics", "swinbert",
        "txt_backbone", "fusion_encoder",
        "txt_backbone_embed_only", "tokenizer"]
    if args.size_epoch == 0:
        toUpdate += ['size_frame', 'size_txt', 'size_img', 'img_transform']
    for key in training_args:
        if key in toUpdate:
            args[key] = training_args[key]
        if "vidswin" in key:
            new_key = key.replace("vidswin", "vis_backbone")
            print(f"Make old key compatible, old: {key}, new {new_key}")
            args[new_key] = training_args[key]
        if "backbone" in key and not (
                'vis_backbone' in key or 'txt_backbone' in key):
            new_key = key.replace("backbone", "vis_backbone")
            print(f"Make old key compatible, old: {key}, new {new_key}")
            if new_key in toUpdate:
                args[new_key] = training_args[key]
    if "vis_backbone" not in training_args or "backbone" not in training_args:
        print(f"Evaluating models without specific backbone,"
              f"revert to default: vidswin")
        args.vis_backbone = "vidswin"
    return args
