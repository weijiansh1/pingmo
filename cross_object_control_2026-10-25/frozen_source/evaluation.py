"""Shared independent evaluation of teachers, students and feedback baselines.

All traces include t=0 and the true terminal sample. Disturbance comparisons
use the same exogenous pulse/initial deviation; a zero controller is only an
uncontrolled-response baseline, not a tuned tracking controller.
"""
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from route_common import BASE,ROOT,read,write,commands,parameters
from src.envs.batched_p_channel import BatchedRollRateEnv,DT_POLICY,COMMAND_SCALE_RAD_S,FORCE_LIMIT_N
from src.envs.roll_rate_commands import RollRateCommandProfile


def scenarios(group='test',duration=8.,include_recovery=True):
    cases=[{'profile':c,'case_id':c.command_id,'category':'command','initial_deg_s':0.,
            'pulse_force_n':0.,'pulse_start_s':2.,'pulse_end_s':2.3} for c in commands(group,duration)]
    if include_recovery:
        for initial in (-5.,5.):
            name=f'initial_{initial:+g}'
            cases.append({'profile':RollRateCommandProfile(name,'step',0.,onset_s=1.,duration_s=duration),
                'case_id':name,'category':'initial_deviation','initial_deg_s':initial,
                'pulse_force_n':0.,'pulse_start_s':2.,'pulse_end_s':2.3})
        for force in (-1.,1.):
            name=f'disturbance_{force:+g}'
            cases.append({'profile':RollRateCommandProfile(name,'step',0.,onset_s=1.,duration_s=duration),
                'case_id':name,'category':'disturbance','initial_deg_s':0.,
                'pulse_force_n':force,'pulse_start_s':2.,'pulse_end_s':2.3})
    return cases


def long_scenarios(duration=30.):
    result=scenarios('test',duration,True)
    return [r for r in result if r['category']!='command'
            or (r['profile'].kind=='step' and abs(r['profile'].amplitude_deg_s)==15)]


class PID:
    """Force-request PID; gains are fixed across plants after source validation."""
    def __init__(self,kp,ki,kd): self.gains=(kp,ki,kd)
    def action(self,env,obs):
        kp,ki,kd=self.gains
        desired=kp*(env.reference-env.state@env.plant.c.squeeze(0))+ki*env.integral-kd*env._output_dot
        return ((desired-env.requested)/env.max_step_n).clamp(-1,1).unsqueeze(-1)


def policy_action(policy,env,observation,teacher=False):
    if policy is None: return observation.new_zeros((env.n_envs,1))
    if isinstance(policy,PID): return policy.action(env,observation)
    if teacher: return policy(observation,env.actor_privileged())[0]
    return policy(observation)


def simulate(policy,row,cases,*,teacher=False,device='cpu'):
    durations={c['profile'].duration_s for c in cases}
    if len(durations)!=1: raise ValueError('All batched scenarios must have one duration')
    duration=durations.pop();steps=round(duration/DT_POLICY)
    env=BatchedRollRateEnv(parameters(row),len(cases),commands=[c['profile'] for c in cases],
        device=device,reference_mode='fixed',episode_steps=steps,auto_reset=False,sample_alignment='same_time')
    env.command_index=torch.arange(len(cases),device=env.device)
    env.set_initial_output(torch.tensor([math.radians(c['initial_deg_s']) for c in cases],device=env.device))
    force=torch.tensor([c['pulse_force_n'] for c in cases],device=env.device)
    starts=torch.tensor([round(c['pulse_start_s']/DT_POLICY) for c in cases],device=env.device)
    ends=torch.tensor([round(c['pulse_end_s']/DT_POLICY) for c in cases],device=env.device)
    tracks={k:[] for k in ('p','reference','command','requested','applied','action','external_force')}
    failed=torch.zeros(len(cases),dtype=torch.bool,device=env.device)
    first_failure=np.full(len(cases),np.nan)
    was_training=getattr(policy,'training',False)
    if hasattr(policy,'eval'): policy.eval()
    def record(action,disturbance):
        values={'p':env.state@env.plant.c.squeeze(0),'reference':env.reference,'command':env.command,
            'requested':env.requested,'applied':env.applied,'action':action.squeeze(-1),'external_force':disturbance}
        for key,value in values.items():
            arr=value.detach().cpu().numpy().copy()
            if key in ('p','requested','applied','action'): arr[failed.cpu().numpy()]=np.nan
            tracks[key].append(arr)
    with torch.no_grad():
        record(torch.zeros(len(cases),1,device=env.device),torch.zeros_like(force))
        for k in range(steps):
            obs=env.observation()
            action=policy_action(policy,env,obs,teacher)
            action=torch.where(failed[:,None],torch.zeros_like(action),action)
            disturbance=force*((k>=starts)&(k<ends))
            _,_,output=env.step(action,external_force_n=disturbance)
            invalid=(~torch.isfinite(output)) | (output.abs()>math.radians(300.))
            new=invalid & ~failed
            first_failure[new.cpu().numpy()]=(k+1)*DT_POLICY
            failed |= invalid
            record(action,disturbance)
            # Censor failed episodes and keep other independent rows running.
            if bool(failed.any()):
                for name in ('state','delay','applied','commanded','requested','integral','history','older','_output_dot'):
                    getattr(env,name)[failed]=0.
    if hasattr(policy,'train'): policy.train(was_training)
    result={k:np.stack(v,axis=1) for k,v in tracks.items()}
    result['time']=np.arange(steps+1)*DT_POLICY
    result['first_failure_s']=first_failure
    return result


def settling(time,error,tolerance,start,min_hold_s=1.):
    """First entry after the last violation; censored unless a tail hold is seen."""
    active=np.flatnonzero(time>=start)
    outside=active[np.abs(error[active])>tolerance]
    candidate=active[0] if not len(outside) else outside[-1]+1
    if candidate>=len(time) or time[-1]-time[candidate]<min_hold_s-1e-9:
        return None,True
    return float(time[candidate]-start),False


def case_metrics(trace,index,case):
    t=trace['time'];p=np.degrees(trace['p'][index]);ref=np.degrees(trace['reference'][index])
    u=trace['requested'][index];applied=trace['applied'][index]
    profile=case['profile'];category=case['category']
    start=profile.onset_s if category=='command' else (case['pulse_start_s'] if category=='disturbance' else 0.)
    release=case['pulse_end_s'] if category=='disturbance' else (0. if category=='initial_deviation' else start)
    active=t>=start
    failed=bool(np.isfinite(trace['first_failure_s'][index]) or not np.isfinite(p).all())
    base={'case_id':case['case_id'],'category':category,'command_kind':profile.kind,
        'duration_s':float(t[-1]),'failed':failed,
        'first_failure_s':float(trace['first_failure_s'][index]) if failed else None,
        'initial_deg_s':case['initial_deg_s'],'pulse_force_n':case['pulse_force_n']}
    if failed:
        return {**base,'passed':False,'rmse_deg_s':None,'peak_deg_s':None,'tv_n_s':None,
            'tail_rms_deg_s':None,'post_release_rms_deg_s':None,'settling_s':None,
            'settling_censored':True,'envelope_last_first_ratio':None,'peak_count':None,
            'overshoot_percent':None,'saturation_fraction':None,'force_rms_n':None}
    e=p-ref;rmse=float(np.sqrt(np.mean(e[active]**2)));peak=float(np.max(np.abs(e[active])))
    active_dt=(np.arange(1,len(t))*DT_POLICY>=start)
    tv=float(np.sum(np.abs(np.diff(u))[active_dt])/(active_dt.sum()*DT_POLICY))
    is_constant=profile.kind=='step'
    tolerance=max(.2,.02*abs(profile.amplitude_deg_s))
    settle,censored=settling(t,e,tolerance,release) if is_constant else (None,None)
    residual=(p if category!='command' else e)[t>=release]
    peaks,_=find_peaks(np.abs(residual),prominence=.05,distance=5)
    ratio=float(abs(residual[peaks[-1]])/abs(residual[peaks[0]])) if len(peaks)>=2 else None
    overshoot=None
    if category=='command' and profile.kind=='step' and abs(profile.amplitude_deg_s)>0:
        directed=np.sign(profile.amplitude_deg_s)*p[active]
        overshoot=float(max(0.,directed.max()-abs(profile.amplitude_deg_s))/abs(profile.amplitude_deg_s)*100)
    return {**base,'passed':rmse<=1.5 and peak<=5. and tv<=12.,'rmse_deg_s':rmse,'peak_deg_s':peak,
        'tv_n_s':tv,'tail_rms_deg_s':float(np.sqrt(np.mean(e[t>=t[-1]-2.]**2))),
        'post_release_rms_deg_s':float(np.sqrt(np.mean(p[t>=release]**2))) if category!='command' else None,
        'settling_s':settle,'settling_censored':censored,'envelope_last_first_ratio':ratio,
        'peak_count':len(peaks),'overshoot_percent':overshoot,
        'saturation_fraction':float(np.mean(np.abs(u[active])>=21.99)),
        'force_rms_n':float(np.sqrt(np.mean(applied[active]**2)))}


def evaluate(policy,row,cases,*,teacher=False,device='cpu',save_trace=None):
    trace=simulate(policy,row,cases,teacher=teacher,device=device)
    metrics=[{**case_metrics(trace,i,c),'plant_id':row['plant_id']} for i,c in enumerate(cases)]
    if save_trace is not None:
        Path(save_trace).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(save_trace,**trace,case_ids=np.asarray([c['case_id'] for c in cases]))
    return metrics


def aggregate(rows):
    result={'episodes':len(rows),'failed_episodes':sum(r['failed'] for r in rows),
            'passed_episodes':sum(r['passed'] for r in rows),
            'pass_rate':float(np.mean([r['passed'] for r in rows])) if rows else None}
    for key in ('rmse_deg_s','peak_deg_s','tv_n_s','tail_rms_deg_s','post_release_rms_deg_s','settling_s'):
        values=[r[key] for r in rows if r.get(key) is not None]
        result[key+'_median_finite']=float(np.median(values)) if values else None
    return result
