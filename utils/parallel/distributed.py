import torch
import torch.distributed as dist

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('未检测到分布式环境，使用单GPU模式')
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
    
    torch.cuda.set_device(args.gpu)
    dist_backend = 'nccl'
    dist.init_process_group(
        backend=dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    return args