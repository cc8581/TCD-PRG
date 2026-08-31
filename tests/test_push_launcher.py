"""The public train.py entry must dispatch PUSH independently of A/B training."""
import json
import sys
from pathlib import Path

import pytest
import train


def local_paths(tmp_path, *, with_checkpoint=True):
    data = {}
    for name in ('dataset_root', 'acronym_root', 'functional_region_root',
                 'observation_cache_dir', 'cache_index_directory', 'output_root'):
        directory = tmp_path / name
        directory.mkdir()
        data[name] = str(directory)
    if with_checkpoint:
        checkpoint = tmp_path / 'perception.pt'
        checkpoint.write_bytes(b'test fixture')
        data['perception_checkpoint'] = str(checkpoint)
    path = tmp_path / 'paths.yaml'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path, data


def test_push_main_without_perception_and_with_accumulation(tmp_path,monkeypatch):
    paths,local=local_paths(tmp_path,with_checkpoint=False)
    monkeypatch.setattr(sys,'argv',['train.py','--stage','push_evaluator','--paths-config',str(paths),
                                   '--gradient-accumulation-steps','3'])
    calls=[]
    monkeypatch.setattr(train.subprocess,'run',lambda cmd,**kw:calls.append(cmd))
    train.main()
    assert len(calls)==1
    assert 'training.gradient_accumulation_steps=3' in calls[0]
    assert '--perception-checkpoint' not in calls[0]
    assert calls[0][1].endswith('train_push_evaluator.py')


@pytest.mark.parametrize('mode',['resume','pretrain-checkpoint'])
def test_new_checkpoint_modes_are_forwarded(tmp_path,mode):
    paths,_=local_paths(tmp_path,with_checkpoint=False)
    args=train._parse_args(['--stage','push_evaluator','--paths-config',str(paths),'--'+mode,str(tmp_path/'new.pt')])
    command=train._push_evaluator_command(args,tmp_path/'best.pt')
    assert '--'+mode in command
    assert '--perception-checkpoint' not in command


@pytest.mark.parametrize('extra',[['--gpus','2'],['--validate-only','--resume','last.pt'],['--checkpoint-interval','0'],['--perception-checkpoint','old.pt']])
def test_unsupported_push_flags_are_rejected(tmp_path,extra):
    paths,_=local_paths(tmp_path,with_checkpoint=False)
    with pytest.raises(SystemExit):
        train._parse_args(['--stage','push_evaluator','--paths-config',str(paths),*extra])
