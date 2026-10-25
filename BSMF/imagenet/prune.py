#!/usr/bin/env python3
"""Direct structured DSFP channel slicing under simultaneous FLOPs/parameter targets."""
import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
from thop import profile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'source'))
from models.hand_resnet50 import pruned_resnet50
from joint_budget import check_budget, select, profile_cfg
import finetune as training


def build_compact(artifact, config):
    scores = artifact['channel_score_variants']['robust']
    report = select({k: v.detach().cpu().numpy() for k,v in scores.items()}, config)
    state = training.clean_state(artifact['model_state_dict'])
    dense = resnet50(weights=None).eval()
    dense.load_state_dict(state, strict=True)
    compact = pruned_resnet50(cfg=report['cfg'].copy(), dataset='imagenet').eval()
    # There is exactly one index per kept original output channel.
    indices = {}
    for name, module in dense.named_modules():
        if isinstance(module, nn.Conv2d):
            if name in report['layer_counts']:
                k = report['layer_counts'][name]
                rank = torch.argsort(scores[name].cpu(), descending=True, stable=True)
                ix = rank[:k].sort().values
                assert len(ix.unique()) == k and k <= module.out_channels
            else:
                ix = torch.arange(module.out_channels)
            indices[name] = ix
    target = compact.state_dict()
    for name, module in compact.named_modules():
        if isinstance(module, nn.Conv2d):
            if name == 'conv1' or '.downsample.' in name or name.endswith('.conv1'):
                inputs = torch.arange(state[name+'.weight'].shape[1])
            else:
                block, conv = name.rsplit('.',1)
                inputs = indices[block+'.conv'+str(int(conv[-1])-1)]
            weight = state[name+'.weight'].index_select(0,indices[name]).index_select(1,inputs)
            if weight.shape != target[name+'.weight'].shape:
                raise ValueError(f'Channel transfer shape mismatch: {name}')
            target[name+'.weight'] = weight.clone()
        elif isinstance(module, nn.BatchNorm2d):
            conv = ('conv1' if name == 'bn1' else
                    name.replace('.downsample.1','.downsample.0') if '.downsample.1' in name else
                    name.rsplit('.',1)[0]+'.conv'+name[-1])
            for suffix in ('weight','bias','running_mean','running_var'):
                target[name+'.'+suffix] = state[name+'.'+suffix][indices[conv]].clone()
            target[name+'.num_batches_tracked'] = state.get(name+'.num_batches_tracked',torch.tensor(0)).clone()
        elif isinstance(module, nn.Linear):
            target[name+'.weight'] = state[name+'.weight'].clone()
            target[name+'.bias'] = state[name+'.bias'].clone()
    compact.load_state_dict(target, strict=True)
    # Compare against the dense DSF model with corresponding BN outputs zeroed.
    # Tests both selected channel identity and residual/BN transfer semantics.
    handles=[]
    for name in report['layer_counts']:
        block, conv = name.rsplit('.',1)
        bn = dense.get_submodule(block+'.bn'+conv[-1])
        mask = torch.zeros(bn.num_features); mask[indices[name]]=1
        handles.append(bn.register_forward_hook(lambda m,x,y,mask=mask: y*mask.view(1,-1,1,1)))
    with torch.inference_mode():
        x=torch.randn(2,3,224,224)
        a,b=dense(x),compact(x)
        torch.testing.assert_close(a,b,atol=3e-4,rtol=3e-4)
    for handle in handles: handle.remove()
    report['masked_dense_max_abs_error'] = (a-b).abs().max().item()
    probe=copy.deepcopy(compact)
    with torch.inference_mode():
        f,p=profile(probe,inputs=(torch.zeros(1,3,224,224),),verbose=False)
    del probe
    if (int(f),int(p)) != profile_cfg(report['cfg']):
        raise RuntimeError('THOP and analytic resource counts disagree.')
    assert sum(p.numel() for p in compact.parameters())==int(p)
    check_budget(f,p,config)
    report.update(actual_thop_flops=int(f),actual_parameters=int(p),input_shape=[1,3,224,224],
                  score_variant='robust A-row L2 scores; no weight-magnitude/AiB fallback',
                  original_channel_indices={name:ix.tolist() for name,ix in indices.items()},
                  no_pruned_checkpoint_used=True)
    return compact,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    config=json.loads((args.run_dir/'config.json').read_text())
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    artifact=torch.load(args.artifact,map_location='cpu',weights_only=False)
    model,report=build_compact(artifact,config)
    print('JOINT_BUDGET_AND_TRANSFER_PASS',json.dumps({k:report[k] for k in ['pruned_flops','pruned_params','flops_drop_percent','params_drop_percent','masked_dense_max_abs_error']}),flush=True)
    report.update(source_checkpoint=str(args.artifact.resolve()),source_sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
                  source_dsf_top1=float(artifact['prec1']),source_dsf_epoch=int(artifact['epoch']),
                  stage='direct_pruning',preflight=False)
    if torch.cuda.device_count()!=1: raise RuntimeError('Expose exactly one GPU.')
    if args.artifact.resolve().parent != (args.run_dir/'dsf/compress_model').resolve():
        raise RuntimeError('Only this run\'s freshly generated dense DSF checkpoint can initialize pruning.')
    dsf_args=json.loads((args.run_dir/'dsf/config/run_args.json').read_text())['args']
    assert dsf_args['pretrained'] and dsf_args['dsf'] and not dsf_args['sp']
    assert not dsf_args['resume'] and not dsf_args['load_dsf_artifacts']
    official=Path(torch.hub.get_dir())/'checkpoints/resnet50-0676ba61.pth'
    official_hash=hashlib.sha256(official.read_bytes()).hexdigest()
    assert official_hash=='0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'
    report['official_baseline_sha256']=official_hash
    report['official_weights']='torchvision.models.ResNet50_Weights.IMAGENET1K_V1'
    cfg_dir=args.run_dir/'init/config';cfg_dir.mkdir(parents=True,exist_ok=True)
    ckpt_dir=args.run_dir/'init/finetune_model';ckpt_dir.mkdir(parents=True,exist_ok=True)
    training.atomic_json(cfg_dir/'structured_pruning_summary.json',report)
    training.atomic_save(ckpt_dir/'resnet50_structured_pruned_init.pth',{'model_state_dict':model.state_dict(),'cfg':report['cfg'],'provenance':report})
    model=model.cuda().to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _,train_data,val_data=training.get_imagenet(args.data_dir)
    loader_args=SimpleNamespace(workers=args.workers,seed=config['seed'])
    model.train()
    with torch.inference_mode():
        for step,(images,_) in enumerate(training.loader(train_data,loader_args,True)):
            assert images.shape[0]==256
            model(images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last))
            if (step+1)%50==0: print('TRAIN_ONLY_BN_CALIBRATION',step+1,'/300',flush=True)
            if step+1==300: break
    metrics=training.validate(model,training.loader(val_data,loader_args,False))
    # Validate resource budgets, finite values, and weight transfer before recovery.
    if not np.isfinite(metrics['loss']): raise RuntimeError('Nonfinite initialization.')
    training.atomic_save(ckpt_dir/'resnet50_structured_pruned_init_bncalibrated.pth',
                         {'model_state_dict':model.state_dict(),'cfg':report['cfg'],
                          'prec1':metrics['top1'],'prec5':metrics['top5'],'provenance':report})
    training.atomic_json(args.run_dir/'initialization_status.json',{'state':'ready','resources':report,'validation':metrics})
    print('PRUNING_READY',json.dumps(metrics),flush=True)


if __name__=='__main__': main()
