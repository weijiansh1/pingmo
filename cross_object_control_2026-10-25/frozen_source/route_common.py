"""Shared, explicit contracts for the new controller-route experiment."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'pingmo_src'))
from src.aircraft.parameters import PChannelParameters
from src.envs.roll_rate_commands import RollRateCommandProfile
from src.teacher.ppo.networks import PPOTeacherActor

PARAMETERS=('l_fa','lambda_s','t_r','zeta_d','omega_d','r_omega','r_zeta','tau_p')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library():
    path=ROOT/'pingmo_src/data/aircraft/generated/p_channel_library_iv_a_5000_v1/plants.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def parameters(row):
    return PChannelParameters(**{k:row['parameters'][k] for k in PARAMETERS})


def commands(group, duration=8.):
    """Fixed disjoint waveforms; training has 8 per family (32 total)."""
    result=[]
    def add(kind,amplitude=0.,**kwargs):
        result.append(RollRateCommandProfile(f'{group}-{len(result):02d}-{kind}',kind,
            amplitude_deg_s=amplitude,onset_s=1.,duration_s=duration,**kwargs))
    if group=='train':
        for sign in (-1,1):
            for a in (5,10,20,30): add('step',sign*a)
            for a in (10,20):
                for w in (.4,.8): add('doublet',sign*a,segment_duration_s=w)
            for a in (8,16):
                for f in (.3,.7): add('sine',sign*a,frequency_hz=f)
            for scale in (.7,1.):
                for freqs in ((.25,.65,1.1),(.4,.9,1.3)):
                    add('multisine',multisine_components=tuple((sign*scale*a,f) for a,f in zip((6,4,2),freqs)))
    elif group=='validation':
        for sign in (-1,1):
            for a in (12,24): add('step',sign*a)
            for a,w in ((12,.55),(18,.95)): add('doublet',sign*a,segment_duration_s=w)
            for a,f in ((10,.45),(14,.85)): add('sine',sign*a,frequency_hz=f)
            for freqs in ((.2,.55,1.4),(.33,.8,1.2)):
                add('multisine',multisine_components=tuple((sign*a,f) for a,f in zip((5,3,2),freqs)))
    elif group=='test':
        for sign in (-1,1):
            for a in (1,7,15,25,31): add('step',sign*a)
            add('doublet',sign*18,segment_duration_s=.7)
            for a,f in ((11,.5),(9,1.1)): add('sine',sign*a,frequency_hz=f)
            add('multisine',multisine_components=tuple((sign*a,f) for a,f in zip((3,2,1),(.2,.6,1.))))
    else:
        raise ValueError(group)
    return result


def waveform_hash(profile):
    return hashlib.sha256(profile.samples(.02).tobytes()).hexdigest()


class Student(nn.Module):
    """Deployment actor sees only the 176 current/history inputs."""
    def __init__(self,width=32,blocks=1,conv_width=8):
        super().__init__()
        self.architecture={'width':width,'blocks':blocks,'conv_width':conv_width}
        self.policy=PPOTeacherActor(**self.architecture,privileged_dim=0,theta_dim=0)
        self.policy.log_std.requires_grad_(False)

    def forward(self,observation):
        return self.policy(observation,observation.new_empty((observation.shape[0],0)))[0]


def load_teacher(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    cfg=payload['config']
    model=PPOTeacherActor(width=cfg['width'],blocks=cfg['blocks'],conv_width=cfg['conv_width'],
        privileged_dim=payload['actor_privileged_dim'])
    model.load_state_dict(payload['actor'])
    return model.to(device).eval()


def save_student(path,model,metadata):
    torch.save({'architecture':model.architecture,'student':model.state_dict(),'metadata':metadata},path)


def load_student(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=Student(**payload['architecture'])
    model.load_state_dict(payload['student'])
    return model.to(device).eval()
